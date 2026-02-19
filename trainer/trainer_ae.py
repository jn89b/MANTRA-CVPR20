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

from tensorboardX import SummaryWriter
from models.model_encdec import model_encdec
from torch.autograd import Variable
import tqdm


class Trainer:
    def __init__(self, config):
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

        print("Dataset created")

        # -------------------------
        # Settings / Model
        # -------------------------
        self.settings = {
            "batch_size": config.batch_size,
            "use_cuda": config.cuda,
            # legacy fields (not used by our AE, but keep if your code expects them)
            "dim_feature_tracklet": config.past_len * 3,
            "dim_feature_future": config.future_len * 3,
            "dim_embedding_key": config.dim_embedding_key,
            "past_len": config.past_len,
            "future_len": config.future_len,
        }
        self.max_epochs = config.max_epochs

        # model
        self.mem_n2n = model_encdec(self.settings)

        # loss
        self.criterionLoss = nn.MSELoss()

        self.opt = torch.optim.Adam(self.mem_n2n.parameters(), lr=config.learning_rate)
        self.iterations = 0
        self.start_epoch = 0
        self.config = config

        if config.cuda:
            self.criterionLoss = self.criterionLoss.cuda()
            self.mem_n2n = self.mem_n2n.cuda()

        # Best validation tracking
        self.best_val = float("inf")

        # Write details to file
        self.write_details()
        self.file.close()

        # Tensorboard summary: configuration
        self.writer = SummaryWriter(self.folder_tensorboard + self.name_test + "_" + config.info)
        self.writer.add_text("Training Configuration", "model name: {}".format(self.mem_n2n.name_model), 0)
        self.writer.add_text("Training Configuration", "dataset train: {}".format(len(self.data_train)), 0)
        self.writer.add_text("Training Configuration", "dataset val: {}".format(len(self.data_val)), 0)
        self.writer.add_text("Training Configuration", "dataset test: {}".format(len(self.data_test)), 0)
        self.writer.add_text("Training Configuration", "batch_size: {}".format(self.config.batch_size), 0)
        self.writer.add_text("Training Configuration", "learning rate init: {}".format(self.config.learning_rate), 0)
        self.writer.add_text("Training Configuration", "dim_embedding_key: {}".format(self.config.dim_embedding_key), 0)

    def write_details(self):
        """
        Serialize configuration parameters to file.
        """
        self.file.write("points of past track: {}".format(self.config.past_len) + "\n")
        self.file.write("points of future track: {}".format(self.config.future_len) + "\n")
        self.file.write("train size: {}".format(len(self.data_train)) + "\n")
        self.file.write("val size: {}".format(len(self.data_val)) + "\n")
        self.file.write("test size: {}".format(len(self.data_test)) + "\n")
        self.file.write("batch size: {}".format(self.config.batch_size) + "\n")
        self.file.write("learning rate: {}".format(self.config.learning_rate) + "\n")
        self.file.write("embedding dim: {}".format(self.config.dim_embedding_key) + "\n")

    def draw_track(self, past, future, pred=None, index_tracklet=0, num_epoch=0, tag="test"):
        """
        Plot past and future trajectory and save it to tensorboard (XY only).
        """
        fig = plt.figure()
        past_np = past.detach().cpu().numpy()
        fut_np = future.detach().cpu().numpy()

        plt.plot(past_np[:, 0], past_np[:, 1], c="blue", marker="o", markersize=3)
        plt.plot(fut_np[:, 0], fut_np[:, 1], c="green", marker="o", markersize=3)

        if pred is not None:
            pred_np = pred.detach().cpu().numpy()
            plt.plot(pred_np[:, 0], pred_np[:, 1], color="red", linewidth=1, marker="o", markersize=1)

        plt.axis("equal")

        buf = io.BytesIO()
        plt.savefig(buf, format="jpeg")
        buf.seek(0)
        image = Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)

        self.writer.add_image(f"Image_{tag}/track{index_tracklet}", image.squeeze(0), num_epoch)
        plt.close(fig)

    def fit(self):
        """
        Autoencoder training procedure.
        """
        config = self.config

        for epoch in range(self.start_epoch, config.max_epochs):
            print(f"----- Epoch: {epoch}")
            loss = self._train_single_epoch()
            print(f"Loss: {loss}")

            # Always validate each epoch
            dict_metrics_val = self.evaluate(self.val_loader, epoch=epoch + 1, tag="val", draw=False)

            # Tensorboard: LR
            for param_group in self.opt.param_groups:
                self.writer.add_scalar("learning_rate", param_group["lr"], epoch)

            # Tensorboard: val metrics
            self.writer.add_scalar("accuracy_val/eucl_mean", dict_metrics_val["eucl_mean"], epoch)
            self.writer.add_scalar("accuracy_val/Horizon10", dict_metrics_val["horizon10"], epoch)
            self.writer.add_scalar("accuracy_val/Horizon20", dict_metrics_val["horizon20"], epoch)
            self.writer.add_scalar("accuracy_val/Horizon30", dict_metrics_val["horizon30"], epoch)
            self.writer.add_scalar("accuracy_val/Horizon40", dict_metrics_val["horizon40"], epoch)

            # Save best checkpoint by val eucl_mean
            if dict_metrics_val["eucl_mean"] < self.best_val:
                self.best_val = dict_metrics_val["eucl_mean"]
                torch.save(self.mem_n2n.state_dict(), self.folder_test + "best_val.pt")

            # Periodic evaluation on train and test
            eval_every = getattr(config, "eval_every", 1)
            
            if (epoch + 1) % eval_every == 0:
                print("eval on TRAIN dataset")
                dict_metrics_train = self.evaluate(self.train_loader, epoch=epoch + 1, tag="train", draw=False)

                print("eval on TEST dataset")
                dict_metrics_test = self.evaluate(self.test_loader, epoch=epoch + 1, tag="test", draw=True)

                # Tensorboard: train metrics
                self.writer.add_scalar("accuracy_train/eucl_mean", dict_metrics_train["eucl_mean"], epoch)
                self.writer.add_scalar("accuracy_train/Horizon10", dict_metrics_train["horizon10"], epoch)
                self.writer.add_scalar("accuracy_train/Horizon20", dict_metrics_train["horizon20"], epoch)
                self.writer.add_scalar("accuracy_train/Horizon30", dict_metrics_train["horizon30"], epoch)
                self.writer.add_scalar("accuracy_train/Horizon40", dict_metrics_train["horizon40"], epoch)

                # Tensorboard: test metrics
                self.writer.add_scalar("accuracy_test/eucl_mean", dict_metrics_test["eucl_mean"], epoch)
                self.writer.add_scalar("accuracy_test/Horizon10", dict_metrics_test["horizon10"], epoch)
                self.writer.add_scalar("accuracy_test/Horizon20", dict_metrics_test["horizon20"], epoch)
                self.writer.add_scalar("accuracy_test/Horizon30", dict_metrics_test["horizon30"], epoch)
                self.writer.add_scalar("accuracy_test/Horizon40", dict_metrics_test["horizon40"], epoch)

                # Save periodic checkpoint
                # include the epoch and val eucl_mean in the filename for easier tracking of checkpoints
                name_checkpoint = f"model_ae_epoch_{epoch}_{self.name_test}_val{dict_metrics_val['eucl_mean']:.4f}.ckpt"
                torch.save(self.mem_n2n.state_dict(), self.folder_test + name_checkpoint)

                # Tensorboard: model weights histogram
                for name, param in self.mem_n2n.named_parameters():
                    self.writer.add_histogram(name, param.data, epoch)

        # Save final trained model
        name_checkpoint = f"model_ae_final_{self.name_test}.ckpt"
        torch.save(self.mem_n2n.state_dict(), self.folder_test + name_checkpoint)

    def evaluate(self, loader, epoch=0, tag="test", draw=False):
        """
        Evaluate the model.
        Returns dict with metrics:
          eucl_mean: mean over samples of mean distance over horizon
          horizon10/20/30/40: distance at those indices if available else 0
        """
        was_training = self.mem_n2n.training
        self.mem_n2n.eval()

        eucl_mean = 0.0
        horizon10 = 0.0
        horizon20 = 0.0
        horizon30 = 0.0
        horizon40 = 0.0

        with torch.no_grad():
            for step, batch in enumerate(tqdm.tqdm(loader)):
                past = batch["past"]
                future = batch["future"]

                if self.config.cuda:
                    past = past.cuda(non_blocking=True)
                    future = future.cuda(non_blocking=True)

                pred = self.mem_n2n(past, future)

                distances = torch.norm(pred - future, dim=2)  # (B,Tf)
                eucl_mean += torch.sum(torch.mean(distances, dim=1)).item()

                Tf = distances.size(1)
                if Tf >= 10:
                    horizon10 += torch.sum(distances[:, 9]).item()
                if Tf >= 20:
                    horizon20 += torch.sum(distances[:, 19]).item()
                if Tf >= 30:
                    horizon30 += torch.sum(distances[:, 29]).item()
                if Tf >= 40:
                    horizon40 += torch.sum(distances[:, 39]).item()

                if draw and step < getattr(self.config, "num_draw_batches", 20):
                    self.draw_track(
                        past[0],
                        future[0],
                        pred[0],
                        index_tracklet=step,
                        num_epoch=epoch,
                        tag=tag,
                    )

        n = len(loader.dataset)
        dict_metrics = {
            "eucl_mean": eucl_mean / n,
            "horizon10": horizon10 / n,
            "horizon20": horizon20 / n,
            "horizon30": horizon30 / n,
            "horizon40": horizon40 / n,
        }

        if was_training:
            self.mem_n2n.train()

        return dict_metrics

    def _train_single_epoch(self):
        """
        Training loop over the dataset for an epoch.
        """
        self.mem_n2n.train()
        config = self.config
        last_loss = None

        for step, batch in enumerate(tqdm.tqdm(self.train_loader)):
            self.iterations += 1
            past = Variable(batch["past"])
            future = Variable(batch["future"])

            if config.cuda:
                past = past.cuda(non_blocking=True)
                future = future.cuda(non_blocking=True)

            self.opt.zero_grad()
            output = self.mem_n2n(past, future)
            loss = self.criterionLoss(output, future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mem_n2n.parameters(), 1.0, norm_type=2)
            self.opt.step()

            self.writer.add_scalar("loss/loss_total", loss.item(), self.iterations)
            last_loss = loss

        return float(last_loss.item() if last_loss is not None else 0.0)