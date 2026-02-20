import os
import datetime
import json
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset_evasion import MantraJsonDataset3D, mantra_collate_3d
from models.model_memory_IRM import IRMLightning # Your Lightning IRM module
from models.model_controllerMem import ControllerLightning 

class IRMTrainerInterface:
    def __init__(self, config):
        self.config = config
        
        # 1. Folder & File Setup (Restored Legacy Logging)
        self.name_test = str(datetime.datetime.now())[:19].replace(":", "-")
        self.folder_tensorboard = 'runs/runs-IRM/'
        self.folder_test = f'training/training_IRM/{self.name_test}_{config.info}/'
        self.checkpoint_dir = self.folder_test + 'checkpoints/'
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        self.file = open(self.folder_test + "details.txt", "w")
        self.logger = TensorBoardLogger(self.folder_tensorboard, name=f"{self.name_test}_{config.info}")

        # 2. Dataset Creation (3D)
        print('creating dataset...')
        self.data_train = MantraJsonDataset3D(
            data_path=getattr(config, "train_data_path", "data/train/"),
            past_len=config.past_len,
            future_len=config.future_len,
            step_size=getattr(config, "step_size", 1),
            use_ego_frame=True,
            return_dummy_scene=True,
        )
        self.train_loader = DataLoader(
            self.data_train, batch_size=config.batch_size, shuffle=True,
            num_workers=getattr(config, "num_workers_train", 8),
            pin_memory=True, collate_fn=mantra_collate_3d,
        )

        self.data_test = MantraJsonDataset3D(
            data_path=getattr(config, "test_data_path", "data/test/"),
            past_len=config.past_len, future_len=config.future_len,
            return_dummy_scene=True,
        )
        self.test_loader = DataLoader(
            self.data_test, batch_size=config.batch_size, shuffle=False,
            num_workers=getattr(config, "num_workers_eval", 1),
            pin_memory=True, collate_fn=mantra_collate_3d,
        )

        # 3. Model Setup (Loading Pretrained Controller)
        self.settings = {
            "batch_size": config.batch_size,
            "use_cuda": config.cuda,
            "dim_embedding_key": config.dim_embedding_key,
            "num_prediction": config.preds,
            "past_len": config.past_len,
            "future_len": config.future_len,
            "learning_rate": config.learning_rate
        }
        
        print(f"Loading pretrained Controller from {config.model}")
        if not os.path.exists(config.model):
            raise FileNotFoundError(
                f"!!! ERROR: Could not find the pretrained Controller checkpoint at: {config.model}\n"
                "Please check the path in your config or ensure the controller training finished successfully."
            )

        if not os.path.isfile(config.model):
            raise IsADirectoryError(
                f"!!! ERROR: The path {config.model} is a directory, but I need the .ckpt file itself."
            )
        
        # Note: Load as ControllerLightning first to get layers
        self.model_ctrl = ControllerLightning.load_from_checkpoint(
            checkpoint_path=config.model,
            model_pretrained=None, # Not needed if loading full checkpoint
            settings=self.settings,
            strict=True,
        )        
        
        # Initialize IRM with pretrained Controller layers
        self.model = IRMLightning(self.settings, self.model_ctrl)

        # 4. Write Legacy Details
        self.write_details()
        self.file.close()

        # 5. Trainer Configuration
        # In IRM, we evaluate based on euclMean (ADE)
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            monitor="val_eucl_mean", 
            filename="model_IRM-{epoch:02d}-{val_eucl_mean:.4f}",
            save_top_k=5,
            mode="min",
        )

        self.trainer = pl.Trainer(
            accelerator="gpu" if config.cuda else "cpu",
            devices=1,
            max_epochs=config.max_epochs,
            logger=self.logger,
            callbacks=[checkpoint_callback],
            gradient_clip_val=1.0,
            # Restore step_results behavior (check val every X epochs)
            check_val_every_n_epoch=1 
        )

    def on_load_checkpoint(self, checkpoint):
        if "state_dict" in checkpoint:
            # Check if memory keys exist in the IRM's own checkpoint
            if "memory_past" in checkpoint["state_dict"]:
                p_shape = checkpoint["state_dict"]["memory_past"].shape
                f_shape = checkpoint["state_dict"]["memory_fut"].shape
                self.memory_past = torch.empty(p_shape, device=self.device)
                self.memory_fut = torch.empty(f_shape, device=self.device)

    def write_details(self):
        self.file.write(f"points of past track: {self.config.past_len}\n")
        self.file.write(f"points of future track: {self.config.future_len}\n")
        self.file.write(f"train size: {len(self.data_train)}\n")
        self.file.write(f"test size: {len(self.data_test)}\n")
        self.file.write(f"batch size: {self.config.batch_size}\n")
        self.file.write(f"learning rate: {self.config.learning_rate}\n")


    def fit(self):
        # Allow model to access data for _memory_writing logic
        self.trainer.train_data = self.data_train 
        
        # Log metadata to TensorBoard
        writer = self.logger.experiment
        writer.add_text('Config', f'Model: {self.model.name_model}', 0)
        writer.add_text('Config', f'Initial Memory: {len(self.model.memory_past)}', 0)

        print('Starting IRM Training...')
        self.trainer.fit(
            self.model, 
            train_dataloaders=self.train_loader,
            val_dataloaders=self.test_loader
        )