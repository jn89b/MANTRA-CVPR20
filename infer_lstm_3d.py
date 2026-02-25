"""
Inference script for LSTM 3D Trajectory Predictor

Evaluates trained LSTM models on test data and generates visualizations.
"""

import argparse
import os
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict,List,Any
from dataset_evasion import MantraJsonDataset3D, mantra_collate_3d
from models.lstm_3d import LSTM3DTrajectoryPredictor, MultiModalLSTM3D

class LSTMMetrics3D:
    """Metrics calculator for 3D trajectory prediction"""
    
    def __init__(self, future_len=20):
        self.future_len = future_len
        self.ade_per_step = []
        self.all_sample_ades = []
        self.all_sample_fdes = []
    
    def update(self, prediction, ground_truth):
        """
        Calculate displacement errors.
        
        For single-mode:
            prediction: (B, Tf, 3)
            ground_truth: (B, Tf, 3)
        
        For multi-mode:
            prediction: (B, K, Tf, 3)
            ground_truth: (B, Tf, 3)
        """
        if prediction.dim() == 4:
            # Multi-modal: Winner-takes-all
            B, K, Tf, _ = prediction.shape
            future_rep = ground_truth.unsqueeze(1).repeat(1, K, 1, 1)
            distances = torch.norm(prediction - future_rep, dim=3)  # (B, K, Tf)
            
            # Select best mode
            mean_distances = torch.mean(distances, dim=2)  # (B, K)
            best_mod_idx = torch.argmin(mean_distances, dim=1)  # (B,)
            best_errors = distances[torch.arange(len(best_mod_idx)), best_mod_idx]  # (B, Tf)
        else:
            # Single-mode
            distances = torch.norm(prediction - ground_truth, dim=2)  # (B, Tf)
            best_errors = distances
        
        self.ade_per_step.append(best_errors.cpu().numpy())
        self.all_sample_ades.extend(torch.mean(best_errors, dim=1).cpu().numpy())
        self.all_sample_fdes.extend(best_errors[:, -1].cpu().numpy())
        
        return best_errors
    
    def report(self, out_dir):
        """Generate and save metrics report"""
        all_errors = np.concatenate(self.ade_per_step, axis=0)
        
        print(f"\n{'='*60}")
        print(f"LSTM 3D TRAJECTORY PREDICTION METRICS")
        print(f"{'='*60}")
        
        total_mean_ade = np.mean(all_errors)
        total_mean_fde = np.mean([e[-1] for e in all_errors])
        
        # Time horizons
        horizons = {"0.5s": 10, "1.0s": 20}
        for label, step in horizons.items():
            if all_errors.shape[1] >= step:
                ade = np.mean(all_errors[:, :step])
                print(f"ADE @ {label}: {ade:.4f} m")
        
        print(f"Total Mean ADE: {total_mean_ade:.4f} m")
        print(f"Total Mean FDE: {total_mean_fde:.4f} m")
        print(f"Min ADE: {np.min(self.all_sample_ades):.4f} m")
        print(f"Max ADE: {np.max(self.all_sample_ades):.4f} m")
        print(f"Std ADE: {np.std(self.all_sample_ades):.4f} m")
        print(f"{'='*60}\n")
        
        self._plot_error_growth(all_errors, out_dir)
        self._plot_histogram(np.mean(self.all_sample_ades), out_dir)
        self._save_metrics(all_errors, out_dir)
        self._save_to_csv(all_errors, out_dir)
    
    def _plot_error_growth(self, all_errors, out_dir):
        """Plot error growth over time"""
        mean_error = np.mean(all_errors, axis=0)
        std_error = np.std(all_errors, axis=0)
        timesteps = np.arange(1, self.future_len + 1)
        
        plt.figure(figsize=(12, 7))
        plt.plot(timesteps, mean_error, 'b-o', label='Mean ADE', linewidth=2, markersize=5)
        plt.fill_between(timesteps, mean_error - std_error, mean_error + std_error,
                         color='blue', alpha=0.2, label='Std Dev ($\sigma$)')
        plt.title('LSTM Prediction Error Over Forecast Horizon')
        plt.xlabel('Forecast Timestep (0.05s intervals)')
        plt.ylabel('Displacement Error (meters)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'lstm_error_growth.png'), dpi=300)
        plt.close()
        print(f"Saved error growth plot")
    
    def _save_to_csv(self, all_errors, out_dir) -> None:
        mean_error = np.mean(all_errors, axis=0)
        std_error = np.std(all_errors, axis=0)
        timesteps = np.arange(1, self.future_len + 1)
        
        results:Dict[str,Any] = {
            "mean_errors": mean_error.tolist(),
            "std_errors": std_error.tolist(),
            "timesteps": timesteps.tolist()
        }
        
        data = pd.DataFrame(results)
        output:str = out_dir + "/csv_metrics.csv"
        data.to_csv(output, index=False)
    
    def _plot_histogram(self, mean_ade, out_dir):
        """Plot ADE distribution histogram"""
        plt.figure(figsize=(12, 7))
        plt.hist(self.all_sample_ades, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(mean_ade, color='red', linestyle='dashed', linewidth=2,
                    label=f'Mean: {mean_ade:.3f}m')
        plt.title('LSTM ADE Distribution Across Dataset')
        plt.xlabel('Average Displacement Error (meters)')
        plt.ylabel('Frequency (Samples)')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'lstm_ade_histogram.png'), dpi=300)
        plt.close()
        print(f"Saved ADE histogram")
    
    def _save_metrics(self, all_errors, out_dir):
        """Save metrics to file"""
        import pickle
        metrics = {
            'mean_errors': np.mean(all_errors, axis=0).tolist(),
            'std_errors': np.std(all_errors, axis=0).tolist(),
            'all_sample_ades': self.all_sample_ades,
            'all_sample_fdes': self.all_sample_fdes,
            'future_len': self.future_len,
        }
        with open(os.path.join(out_dir, 'lstm_metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)
        print(f"Saved metrics to pickle file")


def plot_3d_trajectory(past, future, pred, save_path, is_multimodal=False):
    """Plot 3D trajectory visualization"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    p = past.cpu().numpy()
    f = future.cpu().numpy()
    
    # Plot past
    ax.plot(p[:, 0], p[:, 1], p[:, 2], 'b-o', label='Past', linewidth=2, markersize=4)
    
    # Plot ground truth
    ax.plot(f[:, 0], f[:, 1], f[:, 2], 'g-x', label='Ground Truth', 
            linewidth=2, markersize=5)
    
    # Plot predictions
    if is_multimodal:
        # pred: (K, Tf, 3)
        pr = pred.cpu().numpy()
        colors = plt.cm.Reds(np.linspace(1, 0.4, pr.shape[0]))
        for k in range(pr.shape[0]):
            label = f'Mode {k+1}' if k == 0 else None
            ax.plot(pr[k, :, 0], pr[k, :, 1], pr[k, :, 2],
                   color=colors[k], alpha=0.6, linestyle='--', linewidth=1.5,
                   label='Predictions' if k == 0 else None)
    else:
        # pred: (Tf, 3)
        pr = pred.cpu().numpy()
        ax.plot(pr[:, 0], pr[:, 1], pr[:, 2], 'r--', label='Prediction',
               linewidth=2, alpha=0.7)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('LSTM 3D Trajectory Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_2d_trajectory(past, future, pred, save_path, is_multimodal=False):
    """Plot top-down 2D trajectory view"""
    plt.figure(figsize=(12, 10))
    
    p = past.cpu().numpy()
    f = future.cpu().numpy()
    
    plt.plot(p[:, 0], p[:, 1], 'b-o', label='Past', linewidth=2, markersize=4)
    plt.plot(f[:, 0], f[:, 1], 'g-x', label='Ground Truth', linewidth=2, markersize=5)
    
    if is_multimodal:
        pr = pred.cpu().numpy()
        colors = plt.cm.Reds(np.linspace(1, 0.4, pr.shape[0]))
        for k in range(pr.shape[0]):
            plt.plot(pr[k, :, 0], pr[k, :, 1], color=colors[k], alpha=0.6,
                    linestyle='--', linewidth=1.5,
                    label='Predictions' if k == 0 else None)
    else:
        pr = pred.cpu().numpy()
        plt.plot(pr[:, 0], pr[:, 1], 'r--', label='Prediction', linewidth=2, alpha=0.7)
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('LSTM Top-Down X-Y View')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Inference for LSTM 3D Trajectory Predictor')
    
    parser.add_argument('--checkpoint', type=str, default="training/training_lstm/single_lstm_2026-02-25_14-17-45/checkpoints/lstm-epoch=09-val_loss=7.2325.ckpt",
                        help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, default="data/test",
                        help='Path to test data directory')
    parser.add_argument('--out_dir', type=str, default='evaluation_results_lstm',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_plots', type=int, default=10,
                        help='Number of sample visualizations to save')
    parser.add_argument('--model_type', type=str, default='single',
                        choices=['single', 'multi'],
                        help='Model type to load')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = f"{args.out_dir}_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"LSTM 3D Trajectory Inference")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test data: {args.test_data}")
    print(f"Output dir: {out_dir}")
    print(f"Model type: {args.model_type}")
    print(f"{'='*60}\n")
    
    # Load model
    print("Loading model...")
    if args.model_type == 'single':
        model = LSTM3DTrajectoryPredictor.load_from_checkpoint(args.checkpoint)
    else:
        model = MultiModalLSTM3D.load_from_checkpoint(args.checkpoint)
    
    model = model.to(device).eval()
    print(f"Model loaded successfully")
    
    # Load dataset
    print(f"Loading test data from: {args.test_data}")
    dataset = MantraJsonDataset3D(
        data_path=args.test_data,
        past_len=model.past_len,
        future_len=model.future_len,
        step_size=1,
        use_ego_frame=True,
        return_dummy_scene=False,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=mantra_collate_3d,
        num_workers=4,
    )
    
    print(f"Test samples: {len(dataset)}")
    
    # Initialize metrics
    metrics = LSTMMetrics3D(future_len=model.future_len)
    
    # Inference loop
    print("\nRunning inference...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Processing")):
            past = batch['past'].to(device)
            future = batch['future'].to(device)
            
            # Generate predictions
            pred = model(past)
            
            # Update metrics
            errors = metrics.update(pred, future)
            
            # Save visualizations
            if i < args.num_plots:
                is_multimodal = args.model_type == 'multi'
                
                # Save 3D plot
                plot_3d_trajectory(
                    past[0], future[0], pred[0],
                    os.path.join(out_dir, f'sample_{i}_3d.png'),
                    is_multimodal=is_multimodal
                )
                
                # Save 2D plot
                plot_2d_trajectory(
                    past[0], future[0], pred[0],
                    os.path.join(out_dir, f'sample_{i}_2d.png'),
                    is_multimodal=is_multimodal
                )
    
    # Generate report
    metrics.report(out_dir)
    
    print(f"\nInference completed!")
    print(f"Results saved to: {out_dir}")


if __name__ == '__main__':
    main()
