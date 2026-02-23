import os
import argparse
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

# Project Imports (Ensure these match your file structure)
from dataset_evasion import MantraJsonDataset3D, mantra_collate_3d
from models.model_memory_IRM import IRMLightning
from models.model_controllerMem import ControllerLightning

# ============================
# 3D Metrics & Uncertainty Class
# ============================
class MantraMetrics3D:
    def __init__(self, future_len=20):
        self.future_len = future_len
        self.ade_per_step = [] # List of (B, Tf) arrays
        self.all_sample_ades = [] # Mean ADE per sample

    def update(self, prediction, ground_truth):
        """
        Calculates Winner-Takes-All 3D Euclidean distances.
        prediction: (B, K, Tf, 3)
        ground_truth: (B, Tf, 3)
        """
        # 1. 3D Euclidean Distance (X, Y, Z)
        future_rep = ground_truth.unsqueeze(1).repeat(1, prediction.shape[1], 1, 1)
        distances = torch.norm(prediction - future_rep, dim=3) # (B, K, Tf)

        # 2. Winner-Takes-All: Select best modality based on full trajectory mean
        mean_distances = torch.mean(distances, dim=2) # (B, K)
        best_mod_idx = torch.argmin(mean_distances, dim=1) # (B,)

        # 3. Extract errors for the winner only
        best_errors = distances[torch.arange(len(best_mod_idx)), best_mod_idx] # (B, Tf)
        
        self.ade_per_step.append(best_errors.cpu().numpy())
        self.all_sample_ades.extend(torch.mean(best_errors, dim=1).cpu().numpy())

        return best_errors 
        
    def report(self, out_dir):
        # (Total_Samples, Tf)
        all_errors = np.concatenate(self.ade_per_step, axis=0) 
        
        print(f"\n{'='*40}\n3D ADE PERFORMANCE\n{'='*40}")
        
        # 1. Calculate Global Mean ADE (avoids "mean of means" bias)
        total_mean_ade = np.mean(all_errors) 
        
        # 2. Horizons
        horizons = {"1.0s": 10, "2.0s": 20}
        for label, step in horizons.items():
            if all_errors.shape[1] >= step:
                # Mean error up to that timestep across all samples
                ade = np.mean(all_errors[:, :step])
                print(f"minADE @ {label}: {ade:.4f} m")
        
        print(f"Total Mean minADE: {total_mean_ade:.4f} m")
        # FDE is just the error at the final index
        print(f"minFDE (Final): {np.mean(all_errors[:, -1]):.4f} m")
        self._plot_error_growth(all_errors, out_dir)
        self.pickle_metrics(out_dir)
        
    def pickle_metrics(self, out_dir:str):
        all_errors = np.concatenate(self.ade_per_step, axis=0)
        mean_errors = np.mean(all_errors, axis=0)
        std_errors = np.std(all_errors, axis=0)
        metrics = {
            "mean_errors": mean_errors.tolist(),
            "std_errors": std_errors.tolist(),
            "all_sample_ades": self.all_sample_ades,
            "future_len": self.future_len
        }
        """Saves ADEs to a pickle file for later analysis."""
        import pickle
        with open(os.path.join(out_dir, "metrics.pkl"), "wb") as f:
            pickle.dump(metrics, f)
        print(f"Saved metrics to {os.path.join(out_dir, 'metrics.pkl')}")

    def _plot_histogram(self, mean_ade, out_dir):
        """Generates ADE distribution histogram."""
        plt.figure(figsize=(10, 6))
        plt.hist(self.all_sample_ades, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(mean_ade, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_ade:.3f}m')
        plt.title("minADE Distribution Across Dataset (3D)")
        plt.xlabel("Average Displacement Error (meters)")
        plt.ylabel("Frequency (Samples)")
        plt.legend(); plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(out_dir, "ade_distribution_histogram.png"))
        plt.close()

    def _plot_error_growth(self, all_errors, out_dir):
        """Generates Mean Error and Std Dev growth plot."""
        mean_error = np.mean(all_errors, axis=0)
        std_error = np.std(all_errors, axis=0)
        timesteps = np.arange(1, self.future_len + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, mean_error, 'b-o', label='Mean ADE', linewidth=2)
        plt.fill_between(timesteps, mean_error - std_error, mean_error + std_error, 
                         color='blue', alpha=0.2, label='Uncertainty ($\sigma$)')
        plt.title("Prediction Error & Uncertainty Over Forecast Horizon")
        plt.xlabel("Forecast Timestep (0.05s intervals)")
        plt.ylabel("Displacement Error (meters)")
        plt.grid(True, linestyle='--', alpha=0.5); plt.legend()
        plt.savefig(os.path.join(out_dir, "error_growth_over_time.png"))
        print("saving plot")
        #plt.show()
        #plt.close()

# ============================
# Plotting Utilities
# ============================
def plot_3d_sample(past, future, pred, save_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    p, f, pr = past.cpu().numpy(), future.cpu().numpy(), pred.cpu().numpy()
    
    ax.plot(p[:, 0], p[:, 1], p[:, 2], 'b-', label='Past', linewidth=2)
    ax.plot(f[:, 0], f[:, 1], f[:, f.shape[1]-1 if len(f.shape)>=1 else 0], 'g-', label='GT', linewidth=2) # Basic GT plot

    colors = plt.cm.Reds(np.linspace(1, 0.4, pr.shape[0]))
    for k in range(pr.shape[0]):
        ax.plot(pr[k, :, 0], pr[k, :, 1], pr[k, :, 2], color=colors[k], alpha=0.5, linestyle='--')
    ax.set_title("UAS 3D Trajectory (MANTRA)"); ax.set_zlabel("Altitude (m)")
    plt.legend(); plt.savefig(save_path); plt.close()

def plot_2d_sample(past, future, pred, save_path):
    plt.figure(figsize=(10, 8))
    p, f, pr = past.cpu().numpy()[:, :2], future.cpu().numpy()[:, :2], pred.cpu().numpy()[:, :, :2]
    plt.plot(p[:, 0], p[:, 1], 'b-o', label='Past', markersize=3)
    plt.plot(f[:, 0], f[:, 1], 'g-x', label='GT', markersize=4)
    colors = plt.cm.Reds(np.linspace(1, 0.4, pr.shape[0]))
    for k in range(pr.shape[0]):
        plt.plot(pr[k, :, 0], pr[k, :, 1], color=colors[k], alpha=0.6, linestyle='--')
    plt.title("Top-Down X-Y View"); plt.grid(True)
    plt.legend(); plt.savefig(save_path); plt.close()

# ============================
# Main Inference Logic
# ============================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Dataset Initialization
    dataset = MantraJsonDataset3D(data_path=args.test_data, past_len=args.past_len, 
                                  future_len=args.future_len, return_dummy_scene=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=mantra_collate_3d)

    # 2. Model Loading (Fixing Positional Argument & Size Mismatch Errors)
    settings = {"dim_embedding_key": args.dim_embedding_key, "num_prediction": args.preds, 
                "past_len": args.past_len, "future_len": args.future_len, "learning_rate": 0.0001}
    
    print(f"Loading IRM Checkpoint: {args.checkpoint}")
    # Note: model_pretrained=None handles the positional arg error; strict=False handles buffer sizes
    model = IRMLightning.load_from_checkpoint(args.checkpoint, settings=settings, 
                                              model_pretrained=None, strict=False).to(device).eval()

    print(f"Memory Bank Verified: {model.memory_past.shape[0]} segments restored")

    metrics = MantraMetrics3D(future_len=args.future_len)

    # 3. Inference Loop
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Running Inference")):
        
            past, future = batch["past"].to(device), batch["future"].to(device)
            # scene = batch.get("scene_one_hot").to(device) if "scene_one_hot" in batch else None
            # Generate multi-modal refined predictions
            import time 
            start_time = time.time()
            pred = model(past) 
            # print("end time", time.time()-start_time)
            errors = metrics.update(pred, future)
            
            if i == 47:
                print("pred", pred)
                print("future", future)
                print("best errors", errors)
            if i < args.num_plots:
                plot_3d_sample(past[0], future[0], pred[0], os.path.join(args.out_dir, f"sample_{i}_3d.png"))
                plot_2d_sample(past[0], future[0], pred[0], os.path.join(args.out_dir, f"sample_{i}_2d.png"))

    # 4. Reporting & Final Plots
    metrics.report(args.out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, 
        default="training/training_IRM/2026-02-22 23-00-00_/checkpoints/model_IRM-epoch=00-val_eucl_mean=0.4007.ckpt", 
        help="IRM checkpoint file")
    test_data = ["data/blue_0_mantra_data","data/red_0_mantra_data", "data/red_1_mantra_data"]
    test = test_data[0]
    parser.add_argument("--test_data", type=str, default=test, help="Test data directory")
    parser.add_argument("--out_dir", type=str, default="evaluation_results_full/", help="Output for plots/histograms")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_plots", type=int, default=100, help="Number of sample pairs (2D & 3D) to save")
    
    parser.add_argument("--past_len", type=int, default=21)
    parser.add_argument("--future_len", type=int, default=20)
    parser.add_argument("--preds", type=int, default=5)
    parser.add_argument("--dim_embedding_key", type=int, default=48)
    # main(parser.parse_args())
    args = parser.parse_args()

    for data in test_data:
        args.test_data = data
        args.out_dir = (
            f"evaluation_results_full/"
            f"{data.split('/')[-1]}_"
            f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )

        main(args)