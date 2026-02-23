import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_all_metrics(results_dir):
    """
    Crawl through results_dir and load all metrics.pkl files.
    
    Returns:
        list: List of dictionaries containing metrics from each pkl file
    """
    metrics_files = []
    results_path = Path(results_dir)
    
    # Find all metrics.pkl files recursively
    for metrics_file in results_path.rglob("metrics.pkl"):
        print(f"Found metrics file: {metrics_file}")
        with open(metrics_file, 'rb') as f:
            metrics = pickle.load(f)
            metrics_files.append({
                'path': str(metrics_file),
                'data': metrics
            })
    
    print(f"\nTotal metrics files found: {len(metrics_files)}")
    return metrics_files


def aggregate_metrics(metrics_list):
    """
    Aggregate metrics from multiple pkl files by averaging mean_errors and std_errors.
    
    Args:
        metrics_list: List of dictionaries with metrics data
        
    Returns:
        dict: Aggregated metrics with averaged values
    """
    if not metrics_list:
        raise ValueError("No metrics files found to aggregate!")
    
    # Extract arrays
    all_mean_errors = []
    all_std_errors = []
    all_sample_ades = []
    future_len = None
    
    for item in metrics_list:
        metrics = item['data']
        all_mean_errors.append(np.array(metrics['mean_errors']))
        all_std_errors.append(np.array(metrics['std_errors']))
        all_sample_ades.extend(metrics['all_sample_ades'])
        
        if future_len is None:
            future_len = metrics['future_len']
        elif future_len != metrics['future_len']:
            print(f"Warning: Inconsistent future_len found! {future_len} vs {metrics['future_len']}")
    
    # Convert to numpy arrays and compute averages
    all_mean_errors = np.array(all_mean_errors)  # Shape: (num_files, future_len)
    all_std_errors = np.array(all_std_errors)    # Shape: (num_files, future_len)
    
    # Average across all files
    aggregated_mean = np.mean(all_mean_errors, axis=0)
    aggregated_std = np.mean(all_std_errors, axis=0)
    
    # Also compute std of the means to show uncertainty across different test sets
    std_of_means = np.std(all_mean_errors, axis=0)
    
    print(f"\n{'='*60}")
    print(f"AGGREGATED METRICS SUMMARY")
    print(f"{'='*60}")
    print(f"Number of test sets aggregated: {len(metrics_list)}")
    print(f"Future prediction length: {future_len}")
    print(f"Overall Mean ADE: {np.mean(aggregated_mean):.4f} m")
    print(f"Overall Mean FDE: {aggregated_mean[-1]:.4f} m")
    print(f"Total samples across all sets: {len(all_sample_ades)}")
    print(f"{'='*60}\n")
    
    return {
        'aggregated_mean': aggregated_mean,
        'aggregated_std': aggregated_std,
        'std_of_means': std_of_means,
        'all_sample_ades': all_sample_ades,
        'future_len': future_len,
        'num_files': len(metrics_list)
    }


def plot_aggregated_error_growth(aggregated_metrics, out_dir):
    """
    Generates aggregated Mean Error and Std Dev growth plot.
    Similar to _plot_error_growth but uses aggregated data.
    """
    mean_error = aggregated_metrics['aggregated_mean']
    std_error = aggregated_metrics['aggregated_std']
    std_of_means = aggregated_metrics['std_of_means']
    future_len = aggregated_metrics['future_len']
    num_files = aggregated_metrics['num_files']
    
    timesteps = np.arange(1, future_len + 1)

    plt.figure(figsize=(12, 7))
    
    # Plot mean error
    plt.plot(timesteps, mean_error, 'b-o', label='Mean ADE (Averaged)', linewidth=2, markersize=5)
    
    # Fill between with average std deviation
    plt.fill_between(timesteps, mean_error - std_error, mean_error + std_error, 
                     color='blue', alpha=0.2, label=f'Avg Uncertainty ($\sigma$)')
    
    # Optional: Add std of means to show variability across test sets
    plt.fill_between(timesteps, mean_error - std_of_means, mean_error + std_of_means, 
                     color='red', alpha=0.15, label=f'Cross-Dataset Variation')
    
    plt.title(f"Aggregated Prediction Error & Uncertainty Over Forecast Horizon\n({num_files} Test Sets)")
    plt.xlabel("Forecast Timestep (0.05s intervals)")
    plt.ylabel("Displacement Error (meters)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    
    # Save figure
    output_path = os.path.join(out_dir, "aggregated_error_growth_over_time.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved aggregated plot to: {output_path}")
    plt.close()


def plot_ade_distribution(aggregated_metrics, out_dir):
    """
    Plot histogram of all sample ADEs across all test sets.
    """
    all_sample_ades = aggregated_metrics['all_sample_ades']
    mean_ade = np.mean(all_sample_ades)
    
    plt.figure(figsize=(12, 7))
    plt.hist(all_sample_ades, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(mean_ade, color='red', linestyle='dashed', linewidth=2, 
                label=f'Overall Mean: {mean_ade:.3f}m')
    plt.title(f"Aggregated minADE Distribution Across All Test Sets (3D)\n({len(all_sample_ades)} Total Samples)")
    plt.xlabel("Average Displacement Error (meters)")
    plt.ylabel("Frequency (Samples)")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    output_path = os.path.join(out_dir, "aggregated_ade_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved aggregated histogram to: {output_path}")
    plt.close()


def plot_per_dataset_comparison(metrics_list, out_dir):
    """
    Plot comparison of error growth across different datasets.
    """
    plt.figure(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_list)))
    
    for idx, item in enumerate(metrics_list):
        metrics = item['data']
        dataset_name = Path(item['path']).parent.name
        mean_errors = np.array(metrics['mean_errors'])
        timesteps = np.arange(1, len(mean_errors) + 1)
        
        plt.plot(timesteps, mean_errors, '-o', color=colors[idx], 
                label=dataset_name, linewidth=2, markersize=4, alpha=0.7)
    
    plt.title("Error Growth Comparison Across Test Sets")
    plt.xlabel("Forecast Timestep (0.05s intervals)")
    plt.ylabel("Mean Displacement Error (meters)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=9)
    
    output_path = os.path.join(out_dir, "per_dataset_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")
    plt.close()


def save_metrics_to_csv(aggregated_metrics, metrics_list, out_dir):
    """
    Save aggregated metrics to CSV files using pandas DataFrames.
    Creates two CSV files:
    1. timestep_metrics.csv - Error metrics at each timestep
    2. summary_metrics.csv - Overall summary statistics
    """
    future_len = aggregated_metrics['future_len']
    
    # 1. Create DataFrame for timestep-wise metrics
    timestep_data = {
        'timestep': np.arange(1, future_len + 1),
        'time_seconds': np.arange(1, future_len + 1) * 0.05,  # 0.05s intervals
        'mean_error_m': aggregated_metrics['aggregated_mean'],
        'avg_std_error_m': aggregated_metrics['aggregated_std'],
        'cross_dataset_std_m': aggregated_metrics['std_of_means']
    }
    
    df_timesteps = pd.DataFrame(timestep_data)
    
    # Add individual dataset metrics as additional columns
    for idx, item in enumerate(metrics_list):
        dataset_name = Path(item['path']).parent.name
        df_timesteps[f'{dataset_name}_mean_error_m'] = np.array(item['data']['mean_errors'])
        df_timesteps[f'{dataset_name}_std_error_m'] = np.array(item['data']['std_errors'])
    
    # Save timestep metrics
    timestep_csv_path = os.path.join(out_dir, "timestep_metrics.csv")
    df_timesteps.to_csv(timestep_csv_path, index=False, float_format='%.6f')
    print(f"Saved timestep metrics to: {timestep_csv_path}")
    
    # 2. Create DataFrame for summary statistics
    summary_data = {
        'metric': [
            'Overall Mean ADE (m)',
            'Overall Mean FDE (m)',
            'Overall Std ADE (m)',
            'Overall Std FDE (m)',
            'Number of Test Sets',
            'Total Samples',
            'Future Prediction Length (steps)',
            'Prediction Horizon (seconds)',
            'Mean ADE @ 1.0s (m)',
            'Mean ADE @ 2.0s (m)'
        ],
        'value': [
            np.mean(aggregated_metrics['aggregated_mean']),
            aggregated_metrics['aggregated_mean'][-1],
            np.mean(aggregated_metrics['aggregated_std']),
            aggregated_metrics['aggregated_std'][-1],
            aggregated_metrics['num_files'],
            len(aggregated_metrics['all_sample_ades']),
            future_len,
            future_len * 0.05,
            np.mean(aggregated_metrics['aggregated_mean'][:10]) if future_len >= 10 else np.nan,
            np.mean(aggregated_metrics['aggregated_mean'][:20]) if future_len >= 20 else np.nan
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    
    # Save summary metrics
    summary_csv_path = os.path.join(out_dir, "summary_metrics.csv")
    df_summary.to_csv(summary_csv_path, index=False, float_format='%.6f')
    print(f"Saved summary metrics to: {summary_csv_path}")
    
    # 3. Create DataFrame for all sample ADEs (optional, can be large)
    df_samples = pd.DataFrame({
        'sample_id': np.arange(len(aggregated_metrics['all_sample_ades'])),
        'ade_m': aggregated_metrics['all_sample_ades']
    })
    
    samples_csv_path = os.path.join(out_dir, "all_sample_ades.csv")
    df_samples.to_csv(samples_csv_path, index=False, float_format='%.6f')
    print(f"Saved all sample ADEs to: {samples_csv_path}")
    
    return df_timesteps, df_summary, df_samples


def main(args):
    """
    Main function to aggregate metrics and generate plots.
    """
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load all metrics files
    print(f"Searching for metrics.pkl files in: {args.results_dir}")
    metrics_list = load_all_metrics(args.results_dir)
    
    if not metrics_list:
        print("No metrics.pkl files found!")
        return
    
    # Aggregate metrics
    aggregated_metrics = aggregate_metrics(metrics_list)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_aggregated_error_growth(aggregated_metrics, args.out_dir)
    plot_ade_distribution(aggregated_metrics, args.out_dir)
    plot_per_dataset_comparison(metrics_list, args.out_dir)
    
    # Save metrics as CSV files
    print("\nSaving metrics to CSV files...")
    save_metrics_to_csv(aggregated_metrics, metrics_list, args.out_dir)
    
    # Save aggregated metrics
    output_pkl = os.path.join(args.out_dir, "aggregated_metrics.pkl")
    with open(output_pkl, 'wb') as f:
        pickle.dump(aggregated_metrics, f)
    print(f"\nSaved aggregated metrics to: {output_pkl}")
    
    print("\n✓ Aggregation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate metrics from multiple evaluation runs")
    parser.add_argument("--results_dir", type=str, 
                        default="evaluation_results_full",
                        help="Directory containing subdirectories with metrics.pkl files")
    parser.add_argument("--out_dir", type=str,
                        default="evaluation_results_aggregated",
                        help="Output directory for aggregated plots and metrics")
    
    args = parser.parse_args()
    main(args)
