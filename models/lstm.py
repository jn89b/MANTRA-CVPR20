from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from traj_pred.full_dataset import FullEgoIndex

# ===========================
#   Losses
# ===========================


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(pred, target)


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        self.mae = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mae(pred, target)


class MultiAngleCosineLoss(nn.Module):
    """
    Cosine-based angular loss for multiple angles represented as sin/cos pairs.

    Expects pred, target of shape (B, T, 6):
      [sinφ, cosφ, sinθ, cosθ, sinψ, cosψ]

    Computes 1 - cos(angle) per angle, then averages over angles, time, and batch.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, T, 6)
        B, T, C = pred.shape
        assert C == 6, f"Expected 6 channels (sin/cos for 3 angles), got {C}"

        # Reshape to (..., num_angles=3, 2) for [sin, cos]
        pred_pairs = pred.view(B, T, 3, 2)    # (B, T, 3, 2)
        targ_pairs = target.view(B, T, 3, 2)  # (B, T, 3, 2)

        # Normalize each sin/cos pair to unit circle
        pred_norm = torch.linalg.norm(
            pred_pairs, dim=-1, keepdim=True)  # (B, T, 3, 1)
        targ_norm = torch.linalg.norm(targ_pairs, dim=-1, keepdim=True)

        pred_unit = pred_pairs / (pred_norm + self.eps)
        targ_unit = targ_pairs / (targ_norm + self.eps)

        cos_delta = (pred_unit * targ_unit).sum(dim=-
                                                1).clamp(-1.0, 1.0)  # (B, T, 3)

        # roll and yaw -> can be mapped together
        # pitch -> is the blacksheep right now  
        # loss = lat_loss + lon_loss 
        # 0 when angles equal; up to 2 when opposite
        loss = 1.0 - cos_delta
        return loss.mean()


# ===========================
#   LSTM Encoder–Decoder
# ===========================

class LSTMEncoderDecoder(LightningModule):
    """
    LSTM encoder–decoder for predicting future attitudes.

    Inputs:
        past_traj: (B, T_past, input_size)
        - should include at least [φ, θ, ψ] at indices FullEgoIndex.PHI/THETA/PSI

    Outputs:
        pred_future: (B, T_future, 6)
        - [sinφ, cosφ, sinθ, cosθ, sinψ, cosψ] for each future timestep
    """

    def __init__(self, config: Dict[str, Any]):
        super(LSTMEncoderDecoder, self).__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config: Dict[str, Any] = config

        # feature dim (e.g. 12 for full ego state)
        self.input_size: int = config['input_size']
        self.hidden_size: int = config['hidden_size']        # e.g. 128
        self.num_layers: int = config['num_layers']          # e.g. 2
        # length of past window
        self.past_len: int = config['past_len']
        # length of future window
        self.future_len: int = config['future_len']

        # Use sin/cos for all three attitudes (roll, pitch, yaw)
        self.use_cos_sin_att: bool = config.get('use_sin_cos', True)

        if self.use_cos_sin_att:
            print("Using sine/cosine representation for attitude angles.")
            # [sin_phi, cos_phi, sin_theta, cos_theta, sin_psi, cos_psi]
            self.output_size: int = 6
        else:
            print("Using raw angle representation for attitude angles.")
            # Raw angles [phi, theta, psi] (not recommended for big maneuvers)
            self.output_size: int = 3

        # Encoder over the past trajectory
        self.encoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # Decoder that feeds on its own outputs (sin/cos or raw angles)
        self.decoder = nn.LSTM(
            input_size=self.output_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.output_size,
        )

        # Loss
        if self.use_cos_sin_att:
            self.criterion = MultiAngleCosineLoss()
            # self.criterion = MAELoss()
        else:
            self.criterion = MAELoss()  # or MSELoss() if you prefer

    # ---------------------------
    #   Forward
    # ---------------------------

    def forward(
        self,
        past_traj: torch.Tensor,
        future_traj: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Args:
            past_traj: (B, T_past, input_size)
            future_traj: (B, T_future, output_size) if using teacher forcing
            teacher_forcing_ratio: prob of using ground-truth at each step

        Returns:
            outputs: (B, T_future, output_size)
        """
        batch_size = past_traj.size(0)
        assert past_traj.dim() == 3, "past_traj must be (B, T_past, input_size)"

        # Encode the past trajectory
        _, (hidden_state, cell_hidden_state) = self.encoder(past_traj)

        # Initialize decoder input from last past attitude
        if self.use_cos_sin_att:
            # last past angles: (B, 3) = [φ, θ, ψ]
            last_angles = past_traj[:, -1, [
                FullEgoIndex.PHI.value,
                FullEgoIndex.THETA.value,
                FullEgoIndex.PSI.value,
            ]]  # (B, 3)

            # build [sinφ, cosφ, sinθ, cosθ, sinψ, cosψ]
            sin_cos_last = torch.stack(
                [
                    torch.sin(last_angles[:, 0]), torch.cos(
                        last_angles[:, 0]),  # roll
                    torch.sin(last_angles[:, 1]), torch.cos(
                        last_angles[:, 1]),  # pitch
                    torch.sin(last_angles[:, 2]), torch.cos(
                        last_angles[:, 2]),  # yaw
                ],
                dim=-1,
            )  # (B, 6)
            decoder_input = sin_cos_last.unsqueeze(1)  # (B, 1, 6)
        else:
            # raw [φ, θ, ψ]
            last_angles = past_traj[:, -1, [
                FullEgoIndex.PHI.value,
                FullEgoIndex.THETA.value,
                FullEgoIndex.PSI.value,
            ]]  # (B, 3)
            decoder_input = last_angles.unsqueeze(1)   # (B, 1, 3)

        outputs = []
        for t in range(self.future_len):
            out, (hidden_state, cell_hidden_state) = self.decoder(
                decoder_input, (hidden_state, cell_hidden_state)
            )
            pred = self.fc(out)

            if self.use_cos_sin_att:
                # we do this to project this back into our unit circle right here
                # idea is to get the [sin,cos] values and sqrt them to a unit length
                pairs = pred.view(pred.size(0), pred.size(1), 3, 2)
                norm = torch.linalg.norm(pairs, dim=-1, keepdim=True).clamp_min(1e-6)
                pred = (pairs / norm).view(pred.size(0), pred.size(1), 6)

            outputs.append(pred)

            if (future_traj is not None) and (torch.rand(1).item() < teacher_forcing_ratio):
                decoder_input = future_traj[:, t:t+1, :]
            else:
                decoder_input = pred

        outputs = torch.cat(outputs, dim=1)  # (B,T_future,C)
        return outputs

    # ---------------------------
    #   Training / Validation
    # ---------------------------

    def _build_future_targets(self, future_full: torch.Tensor) -> torch.Tensor:
        """
        Build the target tensor for attitude prediction from full future state.

        future_full: (B, T_future, A) where A includes [φ, θ, ψ] at FullEgoIndex indices.

        Returns:
            future_traj: (B, T_future, output_size)
        """
        if self.use_cos_sin_att:
            # Extract future [φ, θ, ψ]
            phi_f = future_full[:, :, FullEgoIndex.PHI.value]
            theta_f = future_full[:, :, FullEgoIndex.THETA.value]
            psi_f = future_full[:, :, FullEgoIndex.PSI.value]

            # Build [sinφ, cosφ, sinθ, cosθ, sinψ, cosψ]
            future_traj = torch.stack(
                [
                    torch.sin(phi_f),   torch.cos(phi_f),
                    torch.sin(theta_f), torch.cos(theta_f),
                    torch.sin(psi_f),   torch.cos(psi_f),
                ],
                dim=-1,
            )  # (B, T_future, 6)
        else:
            # Raw [φ, θ, ψ]
            future_traj = future_full[:, :, [
                FullEgoIndex.PHI.value,
                FullEgoIndex.THETA.value,
                FullEgoIndex.PSI.value,
            ]]  # (B, T_future, 3)

        return future_traj

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Expects batch['input_dict'] with:
          - 'obj_trajs': (B, N_agents, T_past, A)
          - 'center_gt_trajs': (B, N_agents, T_future, A)

        We use only the ego agent (index 0).
        """
        inputs = batch['input_dict']

        # (B, N_agents, T_past, A)
        past_traj = inputs['obj_trajs'].float()
        past_traj = past_traj[:, 0, :, :]              # (B, T_past, A)

        # (B, N_agents, T_future, A)
        future_full = inputs['center_gt_trajs'].float()
        future_full = future_full[:, 0, :, :]            # (B, T_future, A)

        future_traj = self._build_future_targets(
            future_full)  # (B, T_future, output_size)

        pred_future = self.forward(
            past_traj=past_traj,
            future_traj=future_traj,
            teacher_forcing_ratio=0.0,
        )

        loss = self.criterion(pred_future, future_traj)
        self.log('train_loss', loss, prog_bar=True,
                 on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        inputs = batch['input_dict']

        past_traj = inputs['obj_trajs'].float()
        past_traj = past_traj[:, 0, :, :]  # (B, T_past, A)

        future_full = inputs['center_gt_trajs'].float()
        future_full = future_full[:, 0, :, :]  # (B, T_future, A)

        future_traj = self._build_future_targets(
            future_full)  # (B, T_future, output_size)

        pred_future = self.forward(
            past_traj=past_traj,
            future_traj=future_traj,
            teacher_forcing_ratio=0.0,
        )

        loss = self.criterion(pred_future, future_traj)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    # ---------------------------
    #   Optimizer / Scheduler
    # ---------------------------

    def configure_optimizers(self) -> Tuple[optim.Optimizer, Any]:
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config['learning_rate'],
            eps=1e-4,
        )

        # You can tune OneCycleLR params to your Trainer setup
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.get('max_lr', 2e-4),
            steps_per_epoch=self.config.get('steps_per_epoch', 1),
            epochs=self.config.get('epochs', 750),
            pct_start=0.02,
            div_factor=100.0,
            final_div_factor=10.0,
        )

        return [optimizer], [scheduler]
