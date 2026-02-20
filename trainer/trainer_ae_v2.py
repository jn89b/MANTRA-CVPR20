import os
import matplotlib.pyplot as plt
import datetime
import io
from PIL import Image
from torchvision.transforms import ToTensor
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_evasion import MantraJsonDataset3D, mantra_collate_3d
from typing import Dict, Any

from tensorboardX import SummaryWriter
from models.model_encdec import ModelEncoderLightning
from torch.autograd import Variable
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import tqdm



class TrainerInterface:
    def __init__(self, config:Dict[str,Any]):
        """
        The Trainer class handles the training procedure for training the autoencoder.
        :param config: configuration parameters (see train_ae.py)
        """

        # test folder creating
        self.name_test = str(datetime.datetime.now())[:13]
        self.folder_tensorboard = "runs/runs-ae/"
        self.folder_test = "training/training_ae/" + self.name_test + "_" + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + "/"
        self.file = open(self.folder_test + "details.txt", "w")
        
        self.name:str = (
            "autoencoder_"
            + f"{config.future_len}steps"
        )
        self.logger = TensorBoardLogger(self.folder_tensorboard, name=self.name)
        self.checkpoint_dir = self.name + "_checkpoint/"
        checkpoint_callback = ModelCheckpoint(
            monitor="val_eucl_mean",
            dirpath=self.checkpoint_dir,
            filename="best-ae-{epoch:02d}-{val_eucl_mean:.4f}",
            save_top_k=5,
            mode="min",
        )
        self.settings = {
            "batch_size": config.batch_size,
            "use_cuda": config.cuda,
            # legacy fields (not used by our AE, but keep if your code expects them)
            "dim_feature_tracklet": config.past_len * 3,
            "dim_feature_future": config.future_len * 3,
            "dim_embedding_key": config.dim_embedding_key,
            "past_len": config.past_len,
            "future_len": config.future_len,
            "learning_rate": config.learning_rate
        }
        self.max_epochs = config.max_epochs
        
        self.model = ModelEncoderLightning(self.settings)
        
        # Resume from latest checkpoint if present
        self.latest_checkpoint = None
        if os.path.exists(self.checkpoint_dir):
            checkpoint_files = sorted(
                [
                    os.path.join(self.checkpoint_dir, f)
                    for f in os.listdir(self.checkpoint_dir)
                    if f.endswith(".ckpt")
                ],
                key=os.path.getmtime,
            )
            if checkpoint_files:
                self.latest_checkpoint = checkpoint_files[-1]
                print(f"Resuming training from checkpoint: {self.latest_checkpoint}")


        print("Creating dataset...")

        # (optional legacy read)
        if getattr(config, "dataset_file", None) is not None and os.path.exists(config.dataset_file):
            _ = json.load(open(config.dataset_file))

        # -------------------------
        # Datasets / Loaders
        # -------------------------
        self.data_train = MantraJsonDataset3D(
            data_path=getattr(config, "train_data_path", "data/train/"),
            past_len=config.past_len,
            future_len=config.future_len,
            step_size=getattr(config, "step_size", 1),
            use_ego_frame=getattr(config, "use_ego_frame", True),
            return_dummy_scene=True,
        )
        self.train_loader = DataLoader(
            self.data_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=getattr(config, "num_workers_train", 8),
            pin_memory=True,
            collate_fn=mantra_collate_3d,
        )

        self.data_val = MantraJsonDataset3D(
            data_path=getattr(config, "val_data_path", "data/val/"),
            past_len=config.past_len,
            future_len=config.future_len,
            step_size=getattr(config, "step_size", 1),
            use_ego_frame=getattr(config, "use_ego_frame", True),
            return_dummy_scene=True,
        )
        self.val_loader = DataLoader(
            self.data_val,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=getattr(config, "num_workers_eval", 1),
            pin_memory=True,
            collate_fn=mantra_collate_3d,
        )

        self.data_test = MantraJsonDataset3D(
            data_path=getattr(config, "test_data_path", "data/test/"),
            past_len=config.past_len,
            future_len=config.future_len,
            step_size=getattr(config, "step_size", 1),
            use_ego_frame=getattr(config, "use_ego_frame", True),
            return_dummy_scene=True,
        )
        self.test_loader = DataLoader(
            self.data_test,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=getattr(config, "num_workers_eval", 1),
            pin_memory=True,
            collate_fn=mantra_collate_3d,
        )
        
        self.trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=config.max_epochs,
            logger=self.logger,
            callbacks=[checkpoint_callback],
            gradient_clip_val=1.0,
            precision=16 if torch.cuda.is_available() else 32,
        )

    def fit(self) -> None:
        # Fit
        self.trainer.fit(
            self.model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
            ckpt_path=self.latest_checkpoint,
        )