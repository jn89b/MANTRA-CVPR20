"""
Training script for 3D LSTM Trajectory Predictor

Compatible with MantraJsonDataset3D from dataset_evasion.py
"""

import argparse
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from dataset_evasion import MantraJsonDataset3D, mantra_collate_3d
from models.lstm_3d import LSTM3DTrajectoryPredictor, MultiModalLSTM3D


def parse_args():
    parser = argparse.ArgumentParser(description='Train LSTM for 3D Trajectory Prediction')
    
    # Data parameters
    parser.add_argument('--train_data', type=str, default='data/train',
                        help='Path to training data directory')
    parser.add_argument('--val_data', type=str, default='data/val',
                        help='Path to validation data directory')
    parser.add_argument('--past_len', type=int, default=40,
                        help='Length of past trajectory')
    parser.add_argument('--future_len', type=int, default=60,
                        help='Length of future trajectory')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='single', choices=['single', 'multi'],
                        help='Model type: single (deterministic) or multi (multi-modal)')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--use_velocity', action='store_true',
                        help='Use velocity features (6D input instead of 3D)')
    parser.add_argument('--num_modes', type=int, default=5,
                        help='Number of modes for multi-modal model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5,
                        help='Teacher forcing ratio during training')
    parser.add_argument('--loss_type', type=str, default='euclidean',
                        choices=['euclidean', 'mse', 'mae'],
                        help='Loss function type')
    parser.add_argument('--scheduler', type=str, default='reduce_on_plateau',
                        choices=['onecycle', 'reduce_on_plateau', 'none'],
                        help='Learning rate scheduler')
    
    # Training setup
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--checkpoint_dir', type=str, default='training/training_lstm',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='runs/lstm',
                        help='Directory for tensorboard logs')
    parser.add_argument('--experiment_name', type=str, default='',
                        help='Experiment name for logging')
    
    # Early stopping
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                        help='Early stopping patience')
    
    # Resume training
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def create_dataloaders(args):
    """Create training and validation dataloaders"""
    
    print(f"Loading training data from: {args.train_data}")
    train_dataset = MantraJsonDataset3D(
        data_path=args.train_data,
        past_len=args.past_len,
        future_len=args.future_len,
        step_size=1,
        use_ego_frame=True,
        # return_dummy_scene=False,  # We don't need scene for LSTM
    )
    
    print(f"Loading validation data from: {args.val_data}")
    val_dataset = MantraJsonDataset3D(
        data_path=args.val_data,
        past_len=args.past_len,
        future_len=args.future_len,
        step_size=1,
        use_ego_frame=True,
        #return_dummy_scene=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=mantra_collate_3d,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=mantra_collate_3d,
        pin_memory=True,
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def create_model(args, steps_per_epoch):
    """Create the LSTM model"""
    
    config = {
        'past_len': args.past_len,
        'future_len': args.future_len,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'loss_type': args.loss_type,
        'teacher_forcing_ratio': args.teacher_forcing_ratio,
        'scheduler': args.scheduler,
        'steps_per_epoch': steps_per_epoch,
        'max_epochs': args.max_epochs,
    }
    
    if args.model_type == 'single':
        config['use_velocity'] = args.use_velocity
        model = LSTM3DTrajectoryPredictor(config)
        print(f"Created single-mode LSTM model")
    else:
        config['num_modes'] = args.num_modes
        model = MultiModalLSTM3D(config)
        print(f"Created multi-modal LSTM model with {args.num_modes} modes")
    
    return model


def main():
    args = parse_args()
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_name = args.experiment_name if args.experiment_name else f'{args.model_type}_lstm'
    experiment_dir = os.path.join(args.checkpoint_dir, f'{experiment_name}_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training LSTM 3D Trajectory Predictor")
    print(f"{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Model type: {args.model_type}")
    print(f"Past length: {args.past_len}")
    print(f"Future length: {args.future_len}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"{'='*60}\n")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)
    steps_per_epoch = len(train_loader)
    
    # Create model
    model = create_model(args, steps_per_epoch)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(experiment_dir, 'checkpoints'),
        filename='lstm-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.early_stopping_patience,
        mode='min',
        verbose=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=experiment_name,
        version=timestamp,
    )
    
    # Create trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() and args.gpus > 0 else 'cpu',
        devices=args.gpus if torch.cuda.is_available() and args.gpus > 0 else 1,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Train
    print(f"\nStarting training...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from_checkpoint,
    )
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
