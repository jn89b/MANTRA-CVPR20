# LSTM 3D Trajectory Predictor for MANTRA

## Overview

This implementation provides LSTM-based models for 3D trajectory prediction, designed to work with the `MantraJsonDataset3D` from `dataset_evasion.py`.

## Files Created

1. **`models/lstm_3d.py`** - Core LSTM models
2. **`train_lstm_3d.py`** - Training script
3. **`infer_lstm_3d.py`** - Inference and evaluation script

## Models

### 1. LSTM3DTrajectoryPredictor (Single-Mode)

A deterministic encoder-decoder LSTM that predicts a single future trajectory.

**Features:**
- Predicts 3D positions (x, y, z) from past trajectory
- Optional velocity features (can use position + velocity as 6D input)
- Supports teacher forcing during training
- Multiple loss functions: Euclidean, MSE, MAE
- Computes ADE (Average Displacement Error) and FDE (Final Displacement Error)

**Architecture:**
```
Encoder LSTM → Hidden State → Decoder LSTM → FC Layer → (x, y, z)
```

### 2. MultiModalLSTM3D (Multi-Mode)

A multi-modal LSTM that predicts K different future trajectories (similar to MANTRA).

**Features:**
- Predicts K=5 (default) different trajectory modes
- Shared encoder, multiple decoders
- Winner-takes-all loss (uses best prediction)
- Suitable for scenarios with multiple possible futures

**Architecture:**
```
                 ┌─→ Decoder 1 → Head 1 → Trajectory 1
Encoder LSTM ───┼─→ Decoder 2 → Head 2 → Trajectory 2
                 ├─→ Decoder 3 → Head 3 → Trajectory 3
                 └─→ Decoder K → Head K → Trajectory K
```

## Usage

### Training Single-Mode LSTM

```bash
python train_lstm_3d.py \
    --train_data data/train \
    --val_data data/val \
    --model_type single \
    --past_len 21 \
    --future_len 20 \
    --hidden_size 128 \
    --num_layers 2 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --max_epochs 100 \
    --loss_type euclidean \
    --teacher_forcing_ratio 0.5
```

### Training Multi-Modal LSTM

```bash
python train_lstm_3d.py \
    --train_data data/train \
    --val_data data/val \
    --model_type multi \
    --num_modes 5 \
    --past_len 21 \
    --future_len 20 \
    --hidden_size 128 \
    --num_layers 2 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --max_epochs 100
```

### Training with Velocity Features

```bash
python train_lstm_3d.py \
    --train_data data/train \
    --val_data data/val \
    --model_type single \
    --use_velocity \
    --hidden_size 128 \
    --batch_size 32 \
    --max_epochs 100
```

### Inference and Evaluation

```bash
python infer_lstm_3d.py \
    --checkpoint training/training_lstm/checkpoints/lstm-epoch=XX-val_loss=X.XXXX.ckpt \
    --test_data data/test \
    --out_dir evaluation_results_lstm \
    --model_type single \
    --num_plots 20
```

## Training Arguments

### Data Parameters
- `--train_data`: Path to training data directory
- `--val_data`: Path to validation data directory
- `--past_len`: Length of past trajectory (default: 21)
- `--future_len`: Length of future trajectory (default: 20)

### Model Parameters
- `--model_type`: Model type - `single` or `multi` (default: single)
- `--hidden_size`: LSTM hidden size (default: 128)
- `--num_layers`: Number of LSTM layers (default: 2)
- `--dropout`: Dropout rate (default: 0.1)
- `--use_velocity`: Use velocity features (flag)
- `--num_modes`: Number of modes for multi-modal (default: 5)

### Training Parameters
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay (default: 0.0001)
- `--max_epochs`: Maximum epochs (default: 100)
- `--teacher_forcing_ratio`: Teacher forcing ratio (default: 0.5)
- `--loss_type`: Loss function - `euclidean`, `mse`, or `mae` (default: euclidean)
- `--scheduler`: LR scheduler - `onecycle`, `reduce_on_plateau`, or `none`

### Training Setup
- `--gpus`: Number of GPUs (default: 1)
- `--num_workers`: Data loader workers (default: 4)
- `--checkpoint_dir`: Checkpoint directory (default: training/training_lstm)
- `--early_stopping_patience`: Early stopping patience (default: 20)
- `--resume_from_checkpoint`: Path to checkpoint to resume from

## Inference Arguments

- `--checkpoint`: Path to model checkpoint (required)
- `--test_data`: Path to test data directory (required)
- `--out_dir`: Output directory (default: evaluation_results_lstm)
- `--batch_size`: Batch size (default: 32)
- `--num_plots`: Number of visualizations to save (default: 10)
- `--model_type`: Model type - `single` or `multi`

## Output

### Training Outputs
- Checkpoints saved in `training/training_lstm/[experiment]_[timestamp]/checkpoints/`
- TensorBoard logs in `runs/lstm/[experiment]/[timestamp]/`
- Metrics: train_loss, train_ade, train_fde, val_loss, val_ade, val_fde

### Inference Outputs
- 3D trajectory plots: `sample_X_3d.png`
- 2D trajectory plots: `sample_X_2d.png`
- Error growth plot: `lstm_error_growth.png`
- ADE histogram: `lstm_ade_histogram.png`
- Metrics pickle: `lstm_metrics.pkl`
- Console report with ADE, FDE at different time horizons

## Metrics

The models compute standard trajectory prediction metrics:

- **ADE (Average Displacement Error)**: Mean Euclidean distance across all timesteps
- **FDE (Final Displacement Error)**: Euclidean distance at the final timestep
- **minADE**: Minimum ADE when using multi-modal predictions (winner-takes-all)
- **minFDE**: Minimum FDE when using multi-modal predictions

## Loss Functions

1. **Euclidean Loss** (Recommended): Directly minimizes L2 distance between predictions and ground truth
2. **MSE Loss**: Mean squared error on coordinates
3. **MAE Loss**: Mean absolute error on coordinates

## Key Features

1. **PyTorch Lightning Integration**: Modern training framework with built-in callbacks
2. **Teacher Forcing**: Configurable ratio for training stability
3. **Multi-Modal Support**: Predict multiple possible futures
4. **Velocity Features**: Optional velocity augmentation
5. **Comprehensive Metrics**: ADE, FDE, error growth plots, histograms
6. **3D Visualization**: Both 3D and top-down 2D trajectory plots
7. **Early Stopping**: Automatic training termination
8. **Learning Rate Scheduling**: OneCycleLR and ReduceLROnPlateau options

## Example Workflow

```bash
# 1. Train single-mode LSTM
python train_lstm_3d.py \
    --train_data data/train \
    --val_data data/val \
    --model_type single \
    --max_epochs 100 \
    --experiment_name "baseline_lstm"

# 2. Evaluate on test set
python infer_lstm_3d.py \
    --checkpoint training/training_lstm/baseline_lstm_*/checkpoints/lstm-epoch=XX-val_loss=X.XXXX.ckpt \
    --test_data data/test \
    --model_type single \
    --num_plots 20

# 3. Train multi-modal LSTM
python train_lstm_3d.py \
    --train_data data/train \
    --val_data data/val \
    --model_type multi \
    --num_modes 5 \
    --max_epochs 100 \
    --experiment_name "multimodal_lstm"

# 4. Evaluate multi-modal
python infer_lstm_3d.py \
    --checkpoint training/training_lstm/multimodal_lstm_*/checkpoints/lstm-epoch=XX-val_loss=X.XXXX.ckpt \
    --test_data data/test \
    --model_type multi \
    --num_plots 20
```

## Comparison with MANTRA

| Feature | LSTM (This Implementation) | MANTRA |
|---------|---------------------------|---------|
| Input | 3D positions (x,y,z) | 2D positions + scene raster |
| Memory | LSTM hidden state | Explicit memory bank |
| Multi-modal | Multiple decoders | Iterative Refinement Module |
| Scene context | None (trajectory-only) | Semantic scene map |
| Simplicity |  Simple architecture |  Complex memory mechanism |
| Training time | Fast |  Slower (3-stage) |

## Tips for Best Results

1. **Start simple**: Begin with single-mode model before trying multi-modal
2. **Tune teacher forcing**: 0.5 is a good starting point; reduce if overfitting
3. **Use Euclidean loss**: Better for trajectory prediction than MSE
4. **Monitor both ADE and FDE**: FDE is often more important for planning
5. **Try velocity features**: Can help capture dynamics better
6. **Batch size**: 32-64 works well for most scenarios
7. **Learning rate**: 1e-3 with ReduceLROnPlateau is a safe choice

## Troubleshooting

**High validation loss:**
- Reduce learning rate
- Increase model capacity (hidden_size, num_layers)
- Add velocity features
- Adjust teacher forcing ratio

**Overfitting:**
- Increase dropout
- Add weight decay
- Use data augmentation
- Reduce model size

**Training instability:**
- Enable gradient clipping (already enabled at 1.0)
- Reduce learning rate
- Increase batch size
