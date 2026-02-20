import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, Any
import torch.optim as optim
import pytorch_lightning as pl

class IRMLightning(pl.LightningModule):
    def __init__(self, settings: Dict[str, Any], model_pretrained: nn.Module):
        super().__init__()
        self.save_hyperparameters(settings)
        self.name_model = "MANTRA_IRM"

        # Parameters
        self.dim_embedding_key = self.hparams["dim_embedding_key"]
        self.num_prediction = self.hparams["num_prediction"]
        self.past_len = self.hparams["past_len"]
        self.future_len = self.hparams["future_len"]
        self.coord_dim = 3 

        # Memory Bank Buffers
        self.register_buffer("memory_past", torch.empty(0, self.dim_embedding_key))
        self.register_buffer("memory_fut", torch.empty(0, self.dim_embedding_key))

        # Layers (Re-use pretrained components from Controller)
        self.conv_past = model_pretrained.conv_past
        self.conv_fut = model_pretrained.conv_fut
        self.encoder_past = model_pretrained.encoder_past
        self.encoder_fut = model_pretrained.encoder_fut
        self.decoder = model_pretrained.decoder
        self.FC_output = model_pretrained.FC_output
        self.linear_controller = model_pretrained.linear_controller

        # Freeze Autoencoder layers
        for part in [self.conv_past, self.conv_fut, self.encoder_past, 
                      self.encoder_fut, self.decoder, self.FC_output]:
            for param in part.parameters():
                param.requires_grad = False

        # IRM Scene Processing Layers
        # Input: (B, 4, 64, 64) dummy scene or actual raster
        self.convScene_1 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, stride=2, padding=2), 
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.convScene_2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(), 
            nn.BatchNorm2d(16)
        )
        self.RNN_scene = nn.GRU(16, self.dim_embedding_key, 1, batch_first=True)
        
        # Refinement Head: Predicts 3D offsets for the whole future horizon
        self.fc_refine = nn.Linear(self.dim_embedding_key, self.future_len * self.coord_dim)

        self.relu = nn.ReLU()
        self.criterion_mse = nn.MSELoss()

    # -------------------------------------------------------------------------
    # ORIGINAL LOGIC: ControllerLoss & Memory Management
    # -------------------------------------------------------------------------
    def ControllerLoss(self, prob, sim):
        """ Original MANTRA Loss: prob * sim + (1 - prob) * (1 - sim) """
        loss = prob * sim + (1 - prob) * (1 - sim)
        return torch.sum(loss)

    def init_memory(self, data_train):
        self.memory_past = torch.empty(0, self.dim_embedding_key, device=self.device)
        self.memory_fut = torch.empty(0, self.dim_embedding_key, device=self.device)
        self.eval() 
        with torch.no_grad():
            for _ in range(self.num_prediction + 1):
                j = random.randint(0, len(data_train) - 1)
                sample = data_train[j]
                past, future = sample["past"].unsqueeze(0).to(self.device), sample["future"].unsqueeze(0).to(self.device)
                
                past_t = past.transpose(1, 2)
                story_embed = self.relu(self.conv_past(past_t)).transpose(1, 2)
                _, state_past = self.encoder_past(story_embed)

                fut_t = future.transpose(1, 2)
                fut_embed = self.relu(self.conv_fut(fut_t)).transpose(1, 2)
                _, state_fut = self.encoder_fut(fut_embed)

                self.memory_past = torch.cat((self.memory_past, state_past.squeeze(0)), 0)
                self.memory_fut = torch.cat((self.memory_fut, state_fut.squeeze(0)), 0)
        self.train()

    def _memory_writing(self):
        train_loader = self.trainer.train_dataloader
        self.init_memory(self.trainer.train_data)
        self.eval()
        with torch.no_grad():
            for batch in train_loader:
                self.write_in_memory(batch["past"].to(self.device), batch["future"].to(self.device))
        self.train()

    def write_in_memory(self, past, future):
            """
            Original logic: Decides if past-future is inserted in memory.
            """
            if (self.memory_past.shape[0] < self.num_prediction):
                num_prediction = self.memory_past.shape[0]
            else:
                num_prediction = self.num_prediction

            # Base case: if memory is empty, we must initialize or skip
            if num_prediction == 0: return 

            dim_batch = past.size(0)
            device = self.device
            zero_padding = torch.zeros(1, dim_batch * num_prediction, self.dim_embedding_key * 2, device=device)
            prediction = torch.empty(0, device=device)
            present_temp = past[:, -1].unsqueeze(1)

            # 1. Base Encoding
            past_t = torch.transpose(past, 1, 2)
            story_embed = self.relu(self.conv_past(past_t))
            story_embed = torch.transpose(story_embed, 1, 2)
            _, state_past = self.encoder_past(story_embed)

            # 2. Cosine similarity and memory read
            past_normalized = F.normalize(self.memory_past, p=2, dim=1)
            state_normalized = F.normalize(state_past.squeeze(0), p=2, dim=1)
            weight_read = torch.matmul(past_normalized, state_normalized.transpose(0, 1)).transpose(0, 1)
            index_max = torch.sort(weight_read, descending=True)[1][:, :num_prediction]

            # 3. Base Rollout
            present = present_temp.repeat_interleave(num_prediction, dim=0)
            state_past_repeat = state_past.repeat_interleave(num_prediction, dim=1)
            ind = index_max.flatten()
            info_future = self.memory_fut[ind]
            info_total = torch.cat((state_past_repeat, info_future.unsqueeze(0)), 2)
            
            input_dec = info_total
            state_dec = zero_padding
            for _ in range(self.future_len):
                output_decoder, state_dec = self.decoder(input_dec, state_dec)
                displacement_next = self.FC_output(output_decoder)
                coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
                prediction = torch.cat((prediction, coords_next), 1)
                present = coords_next
                input_dec = zero_padding

            # Reshape to 3D consistency
            prediction = prediction.view(dim_batch, num_prediction, self.future_len, 3)

            # 4. Controller Similarity Check
            future_rep = future.unsqueeze(1).repeat(1, num_prediction, 1, 1)
            distances = torch.norm(prediction - future_rep, dim=3)
            
            tolerance_1s = torch.sum(distances[:, :, :10] < 0.5, dim=2)
            tolerance_2s = torch.sum(distances[:, :, 10:20] < 1, dim=2)
            tolerance_3s = torch.sum(distances[:, :, 20:30] < 1.5, dim=2)
            tolerance_4s = torch.sum(distances[:, :, 30:40] < 2, dim=2)
            
            tolerance = tolerance_1s + tolerance_2s + tolerance_3s + tolerance_4s
            tolerance_rate = (torch.max(tolerance, dim=1)[0].float() / 40).unsqueeze(1)

            # 5. Write Decision
            writing_prob = torch.sigmoid(self.linear_controller(tolerance_rate))

            # Encoding future for memory
            fut_t = torch.transpose(future, 1, 2)
            future_embed = self.relu(self.conv_fut(fut_t))
            future_embed = torch.transpose(future_embed, 1, 2)
            _, state_fut = self.encoder_fut(future_embed)

            index_writing = np.where(writing_prob.detach().cpu().numpy() > 0.5)[0]
            if len(index_writing) > 0:
                past_to_write = state_past.squeeze(0)[index_writing]
                future_to_write = state_fut.squeeze(0)[index_writing]
                self.memory_past = torch.cat((self.memory_past, past_to_write), 0)
                self.memory_fut = torch.cat((self.memory_fut, future_to_write), 0)
                
    def on_train_epoch_start(self):
            print(f"\nEpoch {self.current_epoch}: Population of memory bank...")
            self.init_memory(self.trainer.train_data)
            self.eval()
            with torch.no_grad():
                for batch in self.trainer.train_dataloader:
                    self.write_in_memory(batch["past"].to(self.device), batch["future"].to(self.device))
            self.train()
        
    def forward(self, past, scene=None):
        dim_batch = past.size(0)
        device = past.device
                        
        # If memory is empty, the model cannot function. 
        # Either populate it or return zeros to finish the sanity check.
        if self.memory_past.numel() == 0:
            return torch.zeros(dim_batch, self.num_prediction, self.future_len, 3, device=self.device)

        # 1. Memory Read & Base Decoder
        present_temp = past[:, -1].unsqueeze(1)
        past_t = past.transpose(1, 2)
        story_embed = self.relu(self.conv_past(past_t)).transpose(1, 2)
        _, state_past = self.encoder_past(story_embed)

        past_norm = F.normalize(self.memory_past, p=2, dim=1)
        state_norm = F.normalize(state_past.squeeze(0), p=2, dim=1)
        weight_read = torch.matmul(state_norm, past_norm.t())
        index_max = torch.sort(weight_read, descending=True, dim=1)[1][:, :self.num_prediction]

        # Interleave for multi-modal rollout
        present = present_temp.repeat_interleave(self.num_prediction, dim=0)
        state_past_rep = state_past.repeat_interleave(self.num_prediction, dim=1)
        ind = index_max.flatten()
        info_future = self.memory_fut[ind]
        info_total = torch.cat((state_past_rep, info_future.unsqueeze(0)), 2)

        prediction = torch.empty(0, device=device)
        input_dec, state_dec = info_total, torch.zeros(1, dim_batch * self.num_prediction, self.dim_embedding_key * 2, device=device)
        
        for _ in range(self.future_len):
            output_decoder, state_dec = self.decoder(input_dec, state_dec)
            coords_next = present + self.FC_output(output_decoder).squeeze(0).unsqueeze(1)
            prediction = torch.cat((prediction, coords_next), 1)
            present, input_dec = coords_next, torch.zeros_like(state_dec)

        # # 2. Iterative Refinement Module (IRM)
        # if scene is not None:
        #     # scene: (B, H, W, 4) -> (B, 4, H, W)
        #     scene_feat = self.convScene_1(scene.permute(0, 3, 1, 2))
        #     scene_feat = self.convScene_2(scene_feat).repeat_interleave(self.num_prediction, dim=0)

        #     for _ in range(4):
        #         # Sample 2D spatial features using XY coordinates
        #         # Scale coordinates to [-1, 1] assuming dim_clip=180
        #         indices = 2 * (prediction[:, :, :2] / 180) - 1 
        #         output = F.grid_sample(scene_feat, indices.unsqueeze(1), align_corners=False)
        #         output = output.squeeze(2).permute(0, 2, 1)

        #         output_rnn, _ = self.RNN_scene(output, state_past_rep.transpose(0, 1))
        #         prediction_refine = self.fc_refine(output_rnn[:, -1, :]).view(-1, self.future_len, 3)
        #         prediction = prediction + prediction_refine

        return prediction.view(dim_batch, self.num_prediction, self.future_len, 3)

    def forward_controller(self, past, future):
        """Helper for Controller Loss logic inside training_step."""
        prediction = self.forward(past) # Base predictions (no scene used for writing decision)
        future_rep = future.unsqueeze(1).repeat(1, self.num_prediction, 1, 1)
        dist = torch.norm(prediction - future_rep, dim=3)
        
        # Calculate similarity (sim)
        tol = (torch.sum(dist[:, :, :10] < 0.5, 2) + torch.sum(dist[:, :, 10:20] < 1.0, 2) + 
               torch.sum(dist[:, :, 20:30] < 1.5, 2) + torch.sum(dist[:, :, 30:40] < 2.0, 2))
        sim = (torch.max(tol, dim=1)[0].float() / 40.0).unsqueeze(1)
        
        prob = torch.sigmoid(self.linear_controller(sim))
        return prob, sim

    # -------------------------------------------------------------------------
    # LIGHTNING STEPS
    # -------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        past = batch["past"]
        future = batch["future"]
        # 1. Forward Pass (Refined 3D predictions)
        # Shape: (B, num_prediction, Tf, 3)
        prediction = self(past) 

        # 2. Refinement Loss: Min-ADE (The "Variety Loss")
        future_rep = future.unsqueeze(1).repeat(1, self.num_prediction, 1, 1) # (B, K, Tf, 3)
        distances = torch.norm(prediction - future_rep, dim=3)               # (B, K, Tf)
        
        # Calculate ADE (Average Displacement Error) per modality
        mean_distances = torch.mean(distances, dim=2)  # (B, K)
        
        # KEY FIX: Find the index of the BEST modality for each sample in batch
        # We only backpropagate through the "winner"
        min_ade, index_min = torch.min(mean_distances, dim=1) # (B,)
        
        # Select the best predictions
        best_pred = prediction[torch.arange(past.shape[0]), index_min] # (B, Tf, 3)
        
        # Refinement Loss (MSE only on the best candidate)
        refine_loss = self.criterion_mse(best_pred, future)

        # 3. Controller Loss (Restoring your original logic)
        # Note: We use the base prediction similarity 'sim' here
        prob, sim = self.forward_controller(past, future)
        cont_loss = self.ControllerLoss(prob, sim)

        # Total Loss
        # You can add a weight factor if refine_loss dominates cont_loss
        total_loss = refine_loss + cont_loss
        
        # Logging
        self.log("train/refine_loss", refine_loss, prog_bar=True)
        self.log("train/cont_loss", cont_loss, prog_bar=False)
        self.log("train/total_loss", total_loss, prog_bar=True)
        self.log("mem/size", float(self.memory_past.shape[0]))
        return total_loss

    def validation_step(self, batch, batch_idx):
        past, future = batch["past"], batch["future"]
        pred = self(past) # Shape: (B, num_prediction, Tf, 3)

        # 2. Calculate min-ADE (Original evaluate logic)
        future_rep = future.unsqueeze(1).repeat(1, self.num_prediction, 1, 1)
        distances = torch.norm(pred - future_rep, dim=3)  # (B, k, Tf)
        mean_distances = torch.mean(distances, dim=2)    # (B, k)
        
        # Pick the best prediction out of K for each sample in batch
        min_ade, _ = torch.min(mean_distances, dim=1)
        eucl_mean = torch.mean(min_ade)

        # 3. Log metrics (Restoring Horizon 10s, 20s, etc.)
        self.log("val_eucl_mean", eucl_mean, prog_bar=True, on_epoch=True)
        
        Tf = distances.size(2)
        for h in [10, 20, 30, 40]:
            if Tf >= h:
                # Get distance at horizon h for the BEST prediction modality
                # Note: this matches your original 'horizon10s' logic
                best_idx = torch.argmin(mean_distances, dim=1)
                h_dist = distances[torch.arange(len(best_idx)), best_idx, h-1]
                self.log(f"val_horizon{h}", torch.mean(h_dist), on_epoch=True)

        # 4. Qualitative Plotting Trigger (Restored from Trainer.evaluate)
        if self.hparams.get("save_images") and batch_idx == 0:
            self.draw_track_lightning(batch, pred)
            
        return eucl_mean

    def configure_optimizers(self):
        # 1. Filter only parameters that require grad (Refinement + Controller)
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        optimizer = torch.optim.Adam(
            trainable_params, 
            lr=self.hparams["learning_rate"]
        )
        
        # Restore the ExponentialLR scheduler from your trainer
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

class model_memory_IRM(nn.Module):
    def __init__(self, settings: Dict[str, Any], model_pretrained: nn.Module):
        super(model_memory_IRM, self).__init__()
        self.name_model = 'MANTRA_IRM'

        # parameters
        self.dim_embedding_key = settings["dim_embedding_key"]
        self.num_prediction = settings["num_prediction"]
        self.past_len = settings["past_len"]
        self.future_len = settings["future_len"]
        self.coord_dim = 3  # Updated to 3D

        # similarity criterion
        self.weight_read = []
        self.index_max = []
        self.similarity = nn.CosineSimilarity(dim=1)

        # Memory (Inherited from pretrained)
        self.memory_past = model_pretrained.memory_past
        self.memory_fut = model_pretrained.memory_fut

        # layers (Ported from pretrained)
        self.conv_past = model_pretrained.conv_past
        self.conv_fut = model_pretrained.conv_fut
        self.encoder_past = model_pretrained.encoder_past
        self.encoder_fut = model_pretrained.encoder_fut
        self.decoder = model_pretrained.decoder
        self.FC_output = model_pretrained.FC_output
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.linear_controller = model_pretrained.linear_controller

        # IRM Scene Processing
        # Input shape: (batch, 4 channels, H, W) based on your MantraJsonDataset3D
        self.convScene_1 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, stride=2, padding=2), 
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.convScene_2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(), 
            nn.BatchNorm2d(16)
        )

        self.RNN_scene = nn.GRU(16, self.dim_embedding_key, 1, batch_first=True)

        # Refinement FC layer: Predicts (Tf, 3) offsets
        self.fc_refine = nn.Linear(self.dim_embedding_key, self.future_len * self.coord_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.RNN_scene.weight_ih_l0)
        nn.init.kaiming_normal_(self.RNN_scene.weight_hh_l0)
        nn.init.kaiming_normal_(self.convScene_1[0].weight)
        nn.init.kaiming_normal_(self.convScene_2[0].weight)
        nn.init.kaiming_normal_(self.fc_refine.weight)
        nn.init.zeros_(self.RNN_scene.bias_ih_l0)
        nn.init.zeros_(self.fc_refine.bias)

    def init_memory(self, data_train):
            self.memory_past = torch.empty(0, self.dim_embedding_key, device=self.device)
            self.memory_fut = torch.empty(0, self.dim_embedding_key, device=self.device)
            self.eval() 
            with torch.no_grad():
                for _ in range(self.num_prediction + 1):
                    j = random.randint(0, len(data_train) - 1)
                    sample = data_train[j]
                    past, future = sample["past"].unsqueeze(0).to(self.device), sample["future"].unsqueeze(0).to(self.device)
                    
                    past_t = past.transpose(1, 2)
                    story_embed = self.relu(self.conv_past(past_t)).transpose(1, 2)
                    _, state_past = self.encoder_past(story_embed)
                    
                    fut_t = future.transpose(1, 2)
                    fut_embed = self.relu(self.conv_fut(fut_t)).transpose(1, 2)
                    _, state_fut = self.encoder_fut(fut_embed)

                    self.memory_past = torch.cat((self.memory_past, state_past.squeeze(0)), 0)
                    self.memory_fut = torch.cat((self.memory_fut, state_fut.squeeze(0)), 0)
            self.train()
            
    def forward(self, past, scene=None): 
        """
        Forward pass. Refine predictions generated by MemNet with IRM.
        :param past: past trajectory
        :param scene: surrounding map
        :return: predicted future
        """
        dim_batch = past.size(0)
        device = past.device
        
        # Note: If num_prediction is high, this consumes significant memory
        zero_padding = torch.zeros(1, dim_batch * self.num_prediction, self.dim_embedding_key * 2, device=device)
        prediction = torch.empty(0, device=device)
        present_temp = past[:, -1].unsqueeze(1)

        # 1. Past temporal encoding
        past_t = torch.transpose(past, 1, 2)
        story_embed = self.relu(self.conv_past(past_t))
        story_embed = torch.transpose(story_embed, 1, 2)
        _, state_past = self.encoder_past(story_embed)

        # 2. Cosine similarity and memory read
        past_normalized = F.normalize(self.memory_past, p=2, dim=1)
        state_normalized = F.normalize(state_past.squeeze(0), p=2, dim=1)
        self.weight_read = torch.matmul(state_normalized, past_normalized.t())
        self.index_max = torch.sort(self.weight_read, descending=True, dim=1)[1][:, :self.num_prediction]

        # 3. Base Prediction (MemNet)
        present = present_temp.repeat_interleave(self.num_prediction, dim=0)
        state_past_rep = state_past.repeat_interleave(self.num_prediction, dim=1)
        ind = self.index_max.flatten()

        info_future = self.memory_fut[ind]
        info_total = torch.cat((state_past_rep, info_future.unsqueeze(0)), 2)
        
        input_dec = info_total
        state_dec = zero_padding
        for _ in range(self.future_len):
            output_decoder, state_dec = self.decoder(input_dec, state_dec)
            displacement_next = self.FC_output(output_decoder)
            coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
            prediction = torch.cat((prediction, coords_next), 1)
            present = coords_next
            input_dec = zero_padding

        # 4. Iterative Refinement (IRM)
        # if scene is not None:
        #     # scene expected: (B, H, W, 4) -> (B, 4, H, W)
        #     scene = scene.permute(0, 3, 1, 2)
        #     scene_feat = self.convScene_1(scene)
        #     scene_feat = self.convScene_2(scene_feat)
        #     # Match batch size for num_predictions
        #     scene_feat = scene_feat.repeat_interleave(self.num_prediction, dim=0)

        #     for _ in range(4): # 4 iterations of refinement
        #         # Grid sample requires coordinates in range [-1, 1]
        #         # Assuming scene covers +/- 90 meters (dim_clip=180)
        #         indices = 2 * (prediction[:, :, :2] / 180) - 1 # Only use XY for grid sampling
        #         indices = indices.unsqueeze(1) # (B*k, 1, Tf, 2)

        #         # Sample local scene features along the predicted trajectory
        #         output = F.grid_sample(scene_feat, indices, align_corners=False)
        #         output = output.squeeze(2).permute(0, 2, 1) # (B*k, Tf, 16)

        #         output_rnn, _ = self.RNN_scene(output, state_past_rep.transpose(0,1))
        #         # Predict 3D offsets
        #         prediction_refine = self.fc_refine(output_rnn[:, -1, :]).view(-1, self.future_len, 3)
        #         prediction = prediction + prediction_refine

        prediction = prediction.view(dim_batch, self.num_prediction, self.future_len, 3)
        return prediction