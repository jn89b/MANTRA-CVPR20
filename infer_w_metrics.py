import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
# Need mpl_toolkits for 3D plotting projection to work correctly
from mpl_toolkits.mplot3d import Axes3D 
from tqdm import tqdm
from torch.utils.data import DataLoader

# Project Imports
from dataset_evasion import MantraJsonDataset3D, mantra_collate_3d
from models.model_memory_IRM import IRMLightning
from models.model_controllerMem import ControllerLightning

# ============================
# Metrics Class (No changes needed here)
# ============================
class MantraMetrics3D:
    def __init__(self, future_len=40):
        self.future_len = future_len
        self.ade_per_step = [] 
        self.all_sample_ades = [] 

    def update(self, prediction, ground_truth):
        # Compute 3D Euclidean distances for all K modalities
        future_rep = ground_truth.unsqueeze(1).repeat(1, prediction.shape[1], 1, 1)
        distances = torch.norm(prediction - future_rep, dim=3) # (B, K, Tf)

        # Winner-Takes-All: Select best modality based on mean error
        mean_distances = torch.mean(distances, dim=2) 
        best_mod_idx = torch.argmin(mean_distances, dim=1)

        # Extract errors for the winner only
        best_errors = distances[torch.arange(len(best_mod_idx)), best_mod_idx] 
        
        self.ade_per_step.append(best_errors.cpu().numpy())
        self.all_sample_ades.extend(torch.mean(best_errors, dim=1).cpu().numpy())

    def report(self, out_dir):
        all_errors = np.concatenate(self.ade_per_step, axis=0)
        print(f"\n{'='*40}\n3D ADE PERFORMANCE BY HORIZON\n{'='*40}")
        
        # Assuming 10Hz data
        horizons = {"1.0s": 10, "2.0s": 20, "3.0s": 30, "4.0s": 40}
        for label, step in horizons.items():
            if all_errors.shape[1] >= step:
                ade = np.mean(all_errors[:, :step])
                print(f"minADE @ {label}: {ade:.4f} meters")
        
        mean_ade = np.mean(self.all_sample_ades)
        print(f"Total Mean minADE: {mean_ade:.4f} meters")
        print(f"minFDE (Final): {np.mean(all_errors[:, -1]):.4f} meters")
        print('='*40)

        self.plot_histogram(mean_ade, out_dir)

    def plot_histogram(self, mean_ade, out_dir):
        plt.figure(figsize=(10, 6))
        plt.hist(self.all_sample_ades, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(mean_ade, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_ade:.3f}m')
        plt.title("minADE Distribution Across Dataset (3D)")
        plt.xlabel("Average Displacement Error (meters)")
        plt.ylabel("Frequency (Samples)")
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(out_dir, "ade_distribution_histogram.png"))
        print(f"Histogram saved to {out_dir}/ade_distribution_histogram.png")
        plt.close()

# ============================
# 3D Plotting Utility
# ============================
def plot_3d_sample(past, future, pred, save_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    p, f, pr = past.cpu().numpy(), future.cpu().numpy(), pred.cpu().numpy()
    
    ax.plot(p[:, 0], p[:, 1], p[:, 2], 'b-', label='Past', linewidth=2)
    ax.plot(f[:, 0], f[:, 1], f[:, 2], 'g-', label='Ground Truth', linewidth=2)

    colors = plt.cm.Reds(np.linspace(1, 0.4, pr.shape[0]))
    for k in range(pr.shape[0]):
        ax.plot(pr[k, :, 0], pr[k, :, 1], pr[k, :, 2], 
                color=colors[k], alpha=0.5, linestyle='--')

    ax.set_title("UAS 3D Trajectory Refinement (MANTRA)")
    ax.set_zlabel("Altitude (m)")
    # Set consistent view angle if desired
    # ax.view_init(elev=20., azim=-35)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# ============================
# NEW: 2D Plotting Utility
# ============================
def plot_2d_sample(past, future, pred, save_path):
    """Plots top-down view (X-Y plane) ignoring Z."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    # Select only X and Y dimensions (indices 0 and 1)
    p = past.cpu().numpy()[:, :2]
    f = future.cpu().numpy()[:, :2]
    pr = pred.cpu().numpy()[:, :, :2]
    
    # Plot Past (Blue) and Future (Green) with markers for clarity
    ax.plot(p[:, 0], p[:, 1], 'b-o', label='Past', linewidth=2, markersize=3, alpha=0.7)
    ax.plot(f[:, 0], f[:, 1], 'g-x', label='Ground Truth', linewidth=2, markersize=4)

    # Plot Predictions (Red Gradient)
    colors = plt.cm.Reds(np.linspace(1, 0.4, pr.shape[0]))
    for k in range(pr.shape[0]):
        ax.plot(pr[k, :, 0], pr[k, :, 1], 
                color=colors[k], alpha=0.6, linestyle='--', linewidth=1.5)

    ax.set_title("UAS 2D Trajectory (Top-Down X-Y View)")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    # Crucial for 2D spatial plots to prevent distortion
    ax.axis('equal') 
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# ============================
# Main Execution
# ============================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Dataset Setup
    dataset = MantraJsonDataset3D(
        data_path=args.test_data,
        past_len=args.past_len,
        future_len=args.future_len,
        return_dummy_scene=True
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                            shuffle=False, collate_fn=mantra_collate_3d)

    # 2. Model Loading
    settings = {
        "dim_embedding_key": args.dim_embedding_key,
        "num_prediction": args.preds,
        "past_len": args.past_len,
        "future_len": args.future_len,
        "learning_rate": 0.0001
    }
    
    print(f"Loading trained IRM from: {args.checkpoint}")
    # Correct loading parameters to avoid errors
    model = IRMLightning.load_from_checkpoint(
        args.checkpoint, 
        settings=settings,
        model_pretrained=None, 
        strict=False 
    ).to(device).eval()

    print(f"Verified Memory Bank Size: {model.memory_past.shape[0]} entries")

    metrics = MantraMetrics3D(future_len=args.future_len)
    os.makedirs(args.out_dir, exist_ok=True)

    # 3. Inference Loop
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Running Inference")):
            past = batch["past"].to(device)
            future = batch["future"].to(device)
            scene = batch.get("scene_one_hot").to(device) if "scene_one_hot" in batch else None

            # Prediction: (B, K, Tf, 3)
            pred = model(past, scene)

            metrics.update(pred, future)

            # Generate both plots for the desired number of samples
            if i < args.num_plots:
                # 3D Plot
                plot_3d_sample(past[0], future[0], pred[0], 
                               os.path.join(args.out_dir, f"sample_{i}_3d.png"))
                # 2D Plot
                plot_2d_sample(past[0], future[0], pred[0], 
                               os.path.join(args.out_dir, f"sample_{i}_2d.png"))

    # 4. Final Report & Histogram
    metrics.report(args.out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Update default paths as needed
    parser.add_argument("--checkpoint", type=str, 
        default="/root/coding_projects/MANTRA-CVPR20/training/training_IRM/2026-02-20 02-15-41_/checkpoints/model_IRM-epoch=15-val_eucl_mean=0.2304.ckpt", 
        help="IRM checkpoint file")
    parser.add_argument("--test_data", type=str, default="data/test", help="Test data directory")
    parser.add_argument("--out_dir", type=str, default="evaluation_results_full/", help="Output for plots/histograms")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_plots", type=int, default=10, help="Number of sample pairs (2D & 3D) to save")
    
    parser.add_argument("--past_len", type=int, default=20)
    parser.add_argument("--future_len", type=int, default=20)
    parser.add_argument("--preds", type=int, default=5)
    parser.add_argument("--dim_embedding_key", type=int, default=48)
    
    args = parser.parse_args()
    main(args)