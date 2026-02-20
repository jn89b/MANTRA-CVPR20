import os
import datetime
import json
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset_evasion import MantraJsonDataset3D, mantra_collate_3d
from models.model_controllerMem import ControllerLightning 
from models.model_encdec import ModelEncoderLightning

class TrainerInterface:
    def __init__(self, config):
        self.config = config
        
        # 1. Folder & File Setup (Restored from old code)
        self.name_test = str(datetime.datetime.now())[:19].replace(":", "-")
        self.folder_tensorboard = 'runs/runs-createMem/'
        self.folder_test = f'training/training_controller/{self.name_test}_{config.info}/'
        self.checkpoint_dir = self.folder_test + 'checkpoints/'
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        # Re-opening the details.txt file for logging config params
        self.file = open(self.folder_test + "details.txt", "w")
        
        self.logger = TensorBoardLogger(
            save_dir=self.folder_tensorboard, 
            name=f"{self.name_test}_{config.info}"
        )

        # 2. Dataset Creation
        print('creating dataset...')
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

        self.data_test = MantraJsonDataset3D(
            data_path=getattr(config, "test_data_path", "data/test/"),
            past_len=config.past_len,
            future_len=config.future_len,
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
        print('dataset created')

        # 3. Resume Logic (Restored)
        self.latest_checkpoint = None
        if os.path.exists(self.checkpoint_dir):
            checkpoint_files = [os.path.join(self.checkpoint_dir, f) for f in os.listdir(self.checkpoint_dir) if f.endswith(".ckpt")]
            if checkpoint_files:
                self.latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
                print(f"Resuming from: {self.latest_checkpoint}")

        # 4. Model Setup
        self.settings = {
            "batch_size": config.batch_size,
            "use_cuda": config.cuda,
            "dim_embedding_key": config.dim_embedding_key,
            "num_prediction": config.preds,
            "past_len": config.past_len,
            "future_len": config.future_len,
            "learning_rate": config.learning_rate
        }
        
        self.model_ae = ModelEncoderLightning.load_from_checkpoint(
            checkpoint_path=config.model_ae,
            settings=self.settings
        )        
        self.model = ControllerLightning(self.settings, self.model_ae)

        # 5. Write Details & TensorBoard Config (Restored)
        self.write_details()
        self.file.close()

        # 6. Trainer Configuration
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            monitor="val_loss",
            filename="model_controller-{epoch:02d}-{val_loss:.4f}",
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
        )

    def write_details(self):
        """
        Ported directly from your original code.
        """
        self.file.write('points of past track: {}'.format(self.config.past_len) + '\n')
        self.file.write('points of future track: {}'.format(self.config.future_len) + '\n')
        self.file.write('train size: {}'.format(len(self.data_train)) + '\n')
        self.file.write('test size: {}'.format(len(self.data_test)) + '\n')
        self.file.write('batch size: {}'.format(self.config.batch_size) + '\n')
        self.file.write('learning rate: {}'.format(self.config.learning_rate) + '\n')
        self.file.write('embedding dim: {}'.format(self.config.dim_embedding_key) + '\n')

    def fit(self):
        # Pass dataset reference for the model hook
        self.trainer.train_data = self.data_train 
        
        # Log config to TensorBoard text tab
        writer = self.logger.experiment
        writer.add_text('Training Configuration', f'model name: {self.model.name_model}', 0)
        writer.add_text('Training Configuration', f'dataset train: {len(self.data_train)}', 0)
        writer.add_text('Training Configuration', f'dataset test: {len(self.data_test)}', 0)
        writer.add_text('Training Configuration', f'batch_size: {self.config.batch_size}', 0)
        writer.add_text('Training Configuration', f'learning rate init: {self.config.learning_rate}', 0)

        print('Starting training...')
        self.trainer.fit(
            self.model, 
            train_dataloaders=self.train_loader,
            val_dataloaders=self.test_loader,
            ckpt_path=self.latest_checkpoint
        )