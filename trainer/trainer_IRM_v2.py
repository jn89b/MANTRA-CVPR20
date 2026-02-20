import os
import datetime
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# Project Imports
from dataset_evasion import MantraJsonDataset3D, mantra_collate_3d
from models.model_memory_IRM import IRMLightning 
from models.model_controllerMem import ControllerLightning 

class IRMTrainerInterface:
    def __init__(self, config):
        self.config = config
        
        # 1. Folder & File Setup
        self.name_test = str(datetime.datetime.now())[:19].replace(":", "-")
        self.folder_tensorboard = 'runs/runs-IRM/'
        self.folder_test = f'training/training_IRM/{self.name_test}_{config.info}/'
        self.checkpoint_dir = os.path.join(self.folder_test, 'checkpoints/')
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        self.file = open(os.path.join(self.folder_test, "details.txt"), "w")
        self.logger = TensorBoardLogger(self.folder_tensorboard, name=f"{self.name_test}_{config.info}")

        # 2. Dataset Creation
        self.data_train = MantraJsonDataset3D(
            data_path=getattr(config, "train_data_path", "data/train/"),
            past_len=config.past_len, future_len=config.future_len,
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

        # 3. Model Loading Logic (The Error Fix)
        self.settings = {
            "batch_size": config.batch_size,
            "dim_embedding_key": config.dim_embedding_key,
            "num_prediction": config.preds,
            "past_len": config.past_len,
            "future_len": config.future_len,
            "learning_rate": config.learning_rate
        }
        
        # Check path existence
        if not os.path.isfile(config.model):
            raise FileNotFoundError(f"Checkpoint not found at: {config.model}")

        print(f"Loading pretrained Controller from {config.model}")
        
        # LOAD CONTROLLER
        # We use strict=False to allow the memory buffer to be overwritten
        self.model_ctrl = ControllerLightning.load_from_checkpoint(
            checkpoint_path=config.model,
            model_pretrained=None, # Skeleton mode
            settings=self.settings,
            strict=False 
        )        
        
        # INITIALIZE IRM
        # We pass the loaded controller. IRMLightning __init__ must handle 
        # model_pretrained=None for future reloads.
        self.model = IRMLightning(self.settings, self.model_ctrl)

        # 4. Final Trainer Setup
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
            check_val_every_n_epoch=1 
        )

        self.write_details()
        self.file.close()

    def write_details(self):
        self.file.write(f"points of past track: {self.config.past_len}\n")
        self.file.write(f"points of future track: {self.config.future_len}\n")
        self.file.write(f"train size: {len(self.data_train)}\n")
        self.file.write(f"test size: {len(self.data_test)}\n")
        self.file.write(f"batch size: {self.config.batch_size}\n")

    def fit(self):
        # Crucial for the internal _memory_writing hooks
        self.trainer.train_data = self.data_train 
        
        print(f"Memory bank ready. Size: {len(self.model.memory_past)}")
        self.trainer.fit(self.model, self.train_loader, self.test_loader)