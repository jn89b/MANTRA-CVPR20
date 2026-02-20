import torch
import numpy as np

class MantraMetrics3D:
    def __init__(self, future_len=40):
        self.future_len = future_len
        self.ade_per_step = [] # Will store (N, future_len)

    def update(self, prediction, ground_truth):
        """
        prediction: (B, K, Tf, 3)
        ground_truth: (B, Tf, 3)
        """
        # 1. Compute 3D distances for all K modalities
        # future_rep shape: (B, K, Tf, 3)
        future_rep = ground_truth.unsqueeze(1).repeat(1, prediction.shape[1], 1, 1)
        distances = torch.norm(prediction - future_rep, dim=3) # (B, K, Tf)

        # 2. Winner-Takes-All: Find the best modality per sample based on full ADE
        mean_distances = torch.mean(distances, dim=2) # (B, K)
        best_modality_idx = torch.argmin(mean_distances, dim=1) # (B,)

        # 3. Extract errors for the winner only
        best_distances = distances[torch.arange(len(best_modality_idx)), best_modality_idx] # (B, Tf)
        
        self.ade_per_step.append(best_distances.cpu().numpy())
        return best_distances.mean(dim=1).cpu().numpy()

    def get_horizon_results(self):
        all_errors = np.concatenate(self.ade_per_step, axis=0) # (Total_N, Tf)
        # Horizon metrics (ADE 1s, 2s, 3s, 4s) assuming 10Hz data
        horizons = {
            "ADE_1s": np.mean(all_errors[:, :10]),
            "ADE_2s": np.mean(all_errors[:, :20]),
            "ADE_3s": np.mean(all_errors[:, :30]),
            "ADE_4s": np.mean(all_errors[:, :40]),
            "FDE": np.mean(all_errors[:, -1])
        }
        return horizons