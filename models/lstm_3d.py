"""
LSTM Encoder-Decoder for 3D Trajectory Prediction (X, Y, Z)

Compatible with MantraJsonDataset3D from dataset_evasion.py
Predicts future 3D positions from past trajectory sequences.
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule


# ===========================
#   Losses
# ===========================

class MSELoss(nn.Module):
    """Mean Squared Error Loss for 3D coordinates"""
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(pred, target)


class MAELoss(nn.Module):
    """Mean Absolute Error Loss for 3D coordinates"""
    def __init__(self):
        super(MAELoss, self).__init__()
        self.mae = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mae(pred, target)


class EuclideanLoss(nn.Module):
    """
    Euclidean Distance Loss - computes average L2 distance per timestep.
    Better suited for trajectory prediction as it penalizes displacement directly.
    """
    def __init__(self):
        super(EuclideanLoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: (B, T, 3) - predicted [x, y, z]
        target: (B, T, 3) - ground truth [x, y, z]
        """
        # Compute L2 distance per timestep: (B, T)
        distances = torch.norm(pred - target, dim=-1)
        # Average over batch and time
        return distances.mean()


# ===========================
#   LSTM Encoder–Decoder for 3D Trajectories
# ===========================

class LSTM3DTrajectoryPredictor(LightningModule):
    """
    LSTM encoder–decoder for predicting future 3D trajectories (x, y, z).

    Inputs:
        past_traj: (B, T_past, 3) - [x, y, z] positions

    Outputs:
        pred_future: (B, T_future, 3) - [x, y, z] positions
    """

    def __init__(self, config: Dict[str, Any]):
        super(LSTM3DTrajectoryPredictor, self).__init__()
        self.save_hyperparameters(config)
        self.config: Dict[str, Any] = config

        # Input is 3D position [x, y, z]
        self.input_size: int = 3
        self.output_size: int = 3
        
        self.hidden_size: int = config.get('hidden_size', 128)
        self.num_layers: int = config.get('num_layers', 2)
        self.past_len: int = config['past_len']
        self.future_len: int = config['future_len']
        
        # Optional: Add velocity as additional features
        self.use_velocity: bool = config.get('use_velocity', False)
        if self.use_velocity:
            self.input_size = 6  # [x, y, z, vx, vy, vz]
            print("Using position + velocity features (6D input)")
        else:
            print("Using position only features (3D input)")

        # Encoder over the past trajectory
        self.encoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=config.get('dropout', 0.1) if self.num_layers > 1 else 0.0,
        )

        # Decoder that feeds on its own outputs
        self.decoder = nn.LSTM(
            input_size=self.output_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=config.get('dropout', 0.1) if self.num_layers > 1 else 0.0,
        )

        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.hidden_size // 2, self.output_size),
        )

        # Loss function
        loss_type = config.get('loss_type', 'euclidean')
        if loss_type == 'euclidean':
            self.criterion = EuclideanLoss()
            print("Using Euclidean Distance Loss")
        elif loss_type == 'mse':
            self.criterion = MSELoss()
            print("Using MSE Loss")
        else:
            self.criterion = MAELoss()
            print("Using MAE Loss")

    # ---------------------------
    #   Helper Methods
    # ---------------------------

    def _compute_velocities(self, traj: torch.Tensor) -> torch.Tensor:
        """
        Compute velocities from trajectory positions.
        
        traj: (B, T, 3) - positions
        returns: (B, T, 3) - velocities (first timestep uses forward difference)
        """
        # Forward difference for velocity
        velocities = torch.zeros_like(traj)
        velocities[:, :-1, :] = traj[:, 1:, :] - traj[:, :-1, :]
        # Last timestep copies the previous velocity
        velocities[:, -1, :] = velocities[:, -2, :]
        return velocities

    def _prepare_encoder_input(self, past_traj: torch.Tensor) -> torch.Tensor:
        """
        Prepare encoder input, optionally adding velocity features.
        
        past_traj: (B, T_past, 3)
        returns: (B, T_past, input_size)
        """
        if self.use_velocity:
            velocities = self._compute_velocities(past_traj)
            return torch.cat([past_traj, velocities], dim=-1)  # (B, T, 6)
        return past_traj  # (B, T, 3)

    # ---------------------------
    #   Forward
    # ---------------------------

    def forward(
        self,
        past_traj: torch.Tensor,
        future_traj: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Args:
            past_traj: (B, T_past, 3) - past [x, y, z] positions
            future_traj: (B, T_future, 3) - ground truth future (for teacher forcing)
            teacher_forcing_ratio: probability of using ground truth at each step

        Returns:
            outputs: (B, T_future, 3) - predicted [x, y, z] positions
        """
        batch_size = past_traj.size(0)
        assert past_traj.dim() == 3, "past_traj must be (B, T_past, 3)"
        assert past_traj.size(-1) == 3, "past_traj must have 3 channels (x, y, z)"

        # Prepare encoder input (with velocity if enabled)
        encoder_input = self._prepare_encoder_input(past_traj)

        # Encode the past trajectory
        _, (hidden_state, cell_state) = self.encoder(encoder_input)

        # Initialize decoder input with last past position
        decoder_input = past_traj[:, -1:, :]  # (B, 1, 3)

        outputs = []
        for t in range(self.future_len):
            # Decode one step
            out, (hidden_state, cell_state) = self.decoder(
                decoder_input, (hidden_state, cell_state)
            )
            
            # Project to 3D position
            pred = self.fc(out)  # (B, 1, 3)
            outputs.append(pred)

            # Teacher forcing: use ground truth or prediction
            if (future_traj is not None) and (torch.rand(1).item() < teacher_forcing_ratio):
                decoder_input = future_traj[:, t:t+1, :]
            else:
                decoder_input = pred

        # Concatenate all predictions
        outputs = torch.cat(outputs, dim=1)  # (B, T_future, 3)
        return outputs

    # ---------------------------
    #   Training / Validation
    # ---------------------------

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Expects batch with:
          - 'past': (B, T_past, 3)
          - 'future': (B, T_future, 3)
        """
        past_traj = batch['past'].float()    # (B, T_past, 3)
        future_traj = batch['future'].float()  # (B, T_future, 3)

        # Forward pass with teacher forcing during training
        teacher_forcing_ratio = self.config.get('teacher_forcing_ratio', 0.5)
        pred_future = self.forward(
            past_traj=past_traj,
            future_traj=future_traj,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        # Compute loss
        loss = self.criterion(pred_future, future_traj)
        
        # Additional metrics
        with torch.no_grad():
            # ADE (Average Displacement Error)
            ade = torch.norm(pred_future - future_traj, dim=-1).mean()
            # FDE (Final Displacement Error)
            fde = torch.norm(pred_future[:, -1, :] - future_traj[:, -1, :], dim=-1).mean()
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_ade', ade, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_fde', fde, prog_bar=False, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Expects batch with:
          - 'past': (B, T_past, 3)
          - 'future': (B, T_future, 3)
        """
        past_traj = batch['past'].float()    # (B, T_past, 3)
        future_traj = batch['future'].float()  # (B, T_future, 3)

        # Forward pass without teacher forcing during validation
        pred_future = self.forward(
            past_traj=past_traj,
            future_traj=None,
            teacher_forcing_ratio=0.0,
        )

        # Compute loss
        loss = self.criterion(pred_future, future_traj)
        
        # Compute metrics
        ade = torch.norm(pred_future - future_traj, dim=-1).mean()
        fde = torch.norm(pred_future[:, -1, :] - future_traj[:, -1, :], dim=-1).mean()
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_ade', ade, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_fde', fde, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    # ---------------------------
    #   Optimizer / Scheduler
    # ---------------------------

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config.get('learning_rate', 1e-3),
            weight_decay=self.config.get('weight_decay', 1e-4),
        )

        scheduler_type = self.config.get('scheduler', 'reduce_on_plateau')
        
        if scheduler_type == 'onecycle':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.get('max_lr', 1e-3),
                steps_per_epoch=self.config.get('steps_per_epoch', 100),
                epochs=self.config.get('max_epochs', 100),
                pct_start=0.1,
                div_factor=25.0,
                final_div_factor=1000.0,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                }
            }
        # elif scheduler_type == 'reduce_on_plateau':
        #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #         optimizer,
        #         mode='min',
        #         factor=0.5,
        #         patience=10,
        #         verbose=True,
        #     )
        #     return {
        #         'optimizer': optimizer,
        #         'lr_scheduler': {
        #             'scheduler': scheduler,
        #             'monitor': 'val_loss',
        #             'interval': 'epoch',
        #         }
        #     }
        else:
            return optimizer


# ===========================
#   Multi-Modal LSTM (Bonus)
# ===========================

class MultiModalLSTM3D(LightningModule):
    """
    Multi-modal LSTM that predicts K different future trajectories.
    Similar to MANTRA's multi-modal predictions.
    """

    def __init__(self, config: Dict[str, Any]):
        super(MultiModalLSTM3D, self).__init__()
        self.save_hyperparameters(config)
        self.config = config

        self.num_modes = config.get('num_modes', 5)
        self.hidden_size = config.get('hidden_size', 128)
        self.num_layers = config.get('num_layers', 2)
        self.past_len = config['past_len']
        self.future_len = config['future_len']

        # Shared encoder
        self.encoder = nn.LSTM(
            input_size=3,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=config.get('dropout', 0.1) if self.num_layers > 1 else 0.0,
        )

        # Multiple decoders (one per mode)
        self.decoders = nn.ModuleList([
            nn.LSTM(
                input_size=3,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=config.get('dropout', 0.1) if self.num_layers > 1 else 0.0,
            )
            for _ in range(self.num_modes)
        ])

        # Output heads
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 2, 3),
            )
            for _ in range(self.num_modes)
        ])

        self.criterion = EuclideanLoss()

    def forward(self, past_traj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            past_traj: (B, T_past, 3)

        Returns:
            predictions: (B, num_modes, T_future, 3)
        """
        batch_size = past_traj.size(0)
        
        # Encode
        _, (hidden, cell) = self.encoder(past_traj)

        all_mode_predictions = []
        
        for mode_idx in range(self.num_modes):
            # Copy hidden states for this mode
            h = hidden.clone()
            c = cell.clone()
            
            decoder_input = past_traj[:, -1:, :]  # (B, 1, 3)
            mode_outputs = []
            
            for t in range(self.future_len):
                out, (h, c) = self.decoders[mode_idx](decoder_input, (h, c))
                pred = self.output_heads[mode_idx](out)  # (B, 1, 3)
                mode_outputs.append(pred)
                decoder_input = pred
            
            mode_traj = torch.cat(mode_outputs, dim=1)  # (B, T_future, 3)
            all_mode_predictions.append(mode_traj)
        
        # Stack all modes: (B, K, T_future, 3)
        predictions = torch.stack(all_mode_predictions, dim=1)
        return predictions

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        past_traj = batch['past'].float()
        future_traj = batch['future'].float()

        # Get all mode predictions: (B, K, T_future, 3)
        pred_future = self.forward(past_traj)

        # Winner-takes-all loss: use best mode
        B, K, T, _ = pred_future.shape
        future_expanded = future_traj.unsqueeze(1).expand(-1, K, -1, -1)  # (B, K, T, 3)
        
        # Compute distance for each mode
        distances = torch.norm(pred_future - future_expanded, dim=-1)  # (B, K, T)
        mode_errors = distances.mean(dim=-1)  # (B, K)
        
        # Select best mode per sample
        best_modes = torch.argmin(mode_errors, dim=1)  # (B,)
        best_predictions = pred_future[torch.arange(B), best_modes]  # (B, T, 3)
        
        loss = self.criterion(best_predictions, future_traj)
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        past_traj = batch['past'].float()
        future_traj = batch['future'].float()

        pred_future = self.forward(past_traj)

        # Winner-takes-all
        B, K, T, _ = pred_future.shape
        future_expanded = future_traj.unsqueeze(1).expand(-1, K, -1, -1)
        distances = torch.norm(pred_future - future_expanded, dim=-1)
        mode_errors = distances.mean(dim=-1)
        best_modes = torch.argmin(mode_errors, dim=1)
        best_predictions = pred_future[torch.arange(B), best_modes]
        
        loss = self.criterion(best_predictions, future_traj)
        ade = torch.norm(best_predictions - future_traj, dim=-1).mean()
        fde = torch.norm(best_predictions[:, -1, :] - future_traj[:, -1, :], dim=-1).mean()
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_ade', ade, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_fde', fde, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config.get('learning_rate', 1e-3),
            weight_decay=self.config.get('weight_decay', 1e-4),
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }
