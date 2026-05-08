import argparse

import lightning.pytorch as pl
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from models.unet_precip_regression_lightning import PhysFormerUNet, UNetDSAttention
from utils import dataset_precip


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PrecipGAN(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--learning_rate", type=float, default=5e-5)
        parser.add_argument("--disc_learning_rate", type=float, default=2e-4)
        parser.add_argument("--epochs", type=int, default=30)
        parser.add_argument("--dataset_folder", type=str, default="")
        parser.add_argument("--num_input_images", type=int, default=12)
        parser.add_argument("--num_output_images", type=int, default=1)
        parser.add_argument("--valid_size", type=float, default=0.1)
        parser.add_argument("--use_oversampled_dataset", type=bool, default=True)
        parser.add_argument("--n_channels", type=int, default=12)
        parser.add_argument("--n_classes", type=int, default=1)
        parser.add_argument("--kernels_per_layer", type=int, default=2)
        parser.add_argument("--bilinear", type=bool, default=True)
        parser.add_argument("--reduction_ratio", type=int, default=16)
        parser.add_argument("--transformer_heads", type=int, default=4)
        parser.add_argument("--transformer_layers", type=int, default=1)
        parser.add_argument("--transformer_dropout", type=float, default=0.1)
        parser.add_argument("--physics_threshold", type=float, default=0.3)
        parser.add_argument("--physics_rain_weight", type=float, default=4.0)
        parser.add_argument("--physics_edge_weight", type=float, default=0.1)
        parser.add_argument("--adv_weight", type=float, default=0.01)
        parser.add_argument("--recon_weight", type=float, default=1.0)
        parser.add_argument("--generator_model", type=str, default="PhysFormerUNet")
        parser.add_argument("--init_generator_ckpt", type=str, default=None)
        parser.add_argument("--resume_from_checkpoint", type=str, default=None)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.automatic_optimization = False

        self.generator_cls = self._resolve_generator_class()
        self.generator = self.generator_cls(hparams)
        self.discriminator = PatchDiscriminator(
            in_channels=self.hparams.num_input_images + self.hparams.n_classes
        )

        self.adv_loss = nn.BCEWithLogitsLoss()
        self.recon_loss = nn.MSELoss()

        self.train_dataset = None
        self.valid_dataset = None
        self.train_sampler = None
        self.valid_sampler = None

        self._maybe_load_generator_init()

    def _maybe_load_generator_init(self):
        ckpt_path = getattr(self.hparams, "init_generator_ckpt", None)
        if not ckpt_path:
            return

        loaded = self.generator_cls.load_from_checkpoint(ckpt_path)
        self.generator.load_state_dict(loaded.state_dict(), strict=True)
        print(f"Loaded generator weights from {ckpt_path}")

    def _resolve_generator_class(self):
        generator_model = getattr(self.hparams, "generator_model", "PhysFormerUNet")
        if generator_model in {"PhysFormerUNet", "UNetDS_CoordAtt"}:
            return PhysFormerUNet
        if generator_model == "UNetDSAttention":
            return UNetDSAttention
        raise NotImplementedError(f"Generator model '{generator_model}' not implemented")

    def prepare_data(self):
        precip_dataset = (
            dataset_precip.precipitation_maps_oversampled_h5
            if self.hparams.use_oversampled_dataset
            else dataset_precip.precipitation_maps_h5
        )
        self.train_dataset = precip_dataset(
            in_file=self.hparams.dataset_folder,
            num_input_images=self.hparams.num_input_images,
            num_output_images=self.hparams.num_output_images,
            train=True,
            transform=None,
        )
        self.valid_dataset = precip_dataset(
            in_file=self.hparams.dataset_folder,
            num_input_images=self.hparams.num_input_images,
            num_output_images=self.hparams.num_output_images,
            train=True,
            transform=None,
        )

        num_train = len(self.train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.hparams.valid_size * num_train))

        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.valid_sampler = SubsetRandomSampler(valid_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            sampler=self.train_sampler,
            pin_memory=True,
            num_workers=0,
            persistent_workers=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            sampler=self.valid_sampler,
            pin_memory=True,
            num_workers=0,
            persistent_workers=False,
        )

    def configure_optimizers(self):
        opt_g = optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.5, 0.999),
        )
        opt_d = optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.disc_learning_rate,
            betas=(0.5, 0.999),
        )
        return [opt_g, opt_d]

    def forward(self, x: Tensor) -> Tensor:
        return self.generator(x)

    def _make_disc_input(self, x: Tensor, y: Tensor) -> Tensor:
        if y.dim() == 3:
            y = y.unsqueeze(1)
        return torch.cat([x, y], dim=1)

    def _generator_loss(self, x: Tensor, y: Tensor, y_hat: Tensor):
        fake_logits = self.discriminator(self._make_disc_input(x, y_hat))
        adv = self.adv_loss(fake_logits, torch.ones_like(fake_logits))
        recon = self.recon_loss(y_hat, y.unsqueeze(1))
        total = self.hparams.recon_weight * recon + self.hparams.adv_weight * adv
        return total, recon, adv

    def training_step(self, batch, batch_idx):
        x, y = batch
        opt_g, opt_d = self.optimizers()

        y = y.float()
        y_hat = self.generator(x)

        real_pair = self._make_disc_input(x, y)
        fake_pair = self._make_disc_input(x, y_hat.detach())

        real_logits = self.discriminator(real_pair)
        fake_logits = self.discriminator(fake_pair)
        d_loss_real = self.adv_loss(real_logits, torch.ones_like(real_logits))
        d_loss_fake = self.adv_loss(fake_logits, torch.zeros_like(fake_logits))
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        g_loss, recon_loss, adv_loss = self._generator_loss(x, y, y_hat)
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        self.log("train_g_loss", g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_d_loss", d_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_recon_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_adv_loss", adv_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.generator(x)
        _, recon_loss, adv_loss = self._generator_loss(x, y.float(), y_hat)
        self.log("val_loss", recon_loss, prog_bar=True)
        self.log("val_adv_loss", adv_loss, prog_bar=False)
