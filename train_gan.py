from root import ROOT_DIR
import argparse
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from models.gan_module import PrecipGAN


def find_default_generator_ckpt():
    candidates = [
        ROOT_DIR / "lightning" / "precip_regression" / "PhysFormerUNet",
        ROOT_DIR / "lightning" / "precip_regression" / "UNetDSAttention",
    ]
    for folder in candidates:
        if not folder.exists():
            continue
        ckpts = sorted(
            [path for path in folder.glob("*.ckpt") if "last" not in path.name.lower()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if ckpts:
            return str(ckpts[0])
    return None


def resolve_training_device():
    if not torch.cuda.is_available():
        return "auto", "auto"

    try:
        major, minor = torch.cuda.get_device_capability(0)
        current_arch = f"sm_{major}{minor}"
        supported_arches = set(torch.cuda.get_arch_list())
        if current_arch not in supported_arches:
            print(
                f"[WARN] PyTorch does not support GPU arch {current_arch} on this machine. "
                "Falling back to CPU training."
            )
            return "cpu", 1
    except Exception as exc:
        print(f"[WARN] Failed to validate CUDA compatibility ({exc}). Falling back to CPU training.")
        return "cpu", 1

    return "auto", "auto"


def train_gan(hparams):
    model = PrecipGAN(hparams)

    default_save_path = ROOT_DIR / "lightning" / "precip_gan"

    checkpoint_callback = ModelCheckpoint(
        dirpath=default_save_path,
        filename="GAN_{epoch}-{val_loss:.6f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=default_save_path,
        filename="GAN_last",
        save_top_k=1,
    )
    tb_logger = loggers.TensorBoardLogger(save_dir=default_save_path, name="PrecipGAN")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    accelerator, devices = resolve_training_device()
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=hparams.epochs,
        default_root_dir=default_save_path,
        logger=tb_logger,
        callbacks=[checkpoint_callback, last_checkpoint_callback, lr_monitor],
    )

    trainer.fit(model=model, ckpt_path=hparams.resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default=str(ROOT_DIR / "data" / "CIKM" / "cikm_oversampled_v2.h5"),
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument("--disc_learning_rate", type=float, default=0.0002)
    parser.add_argument("--epochs", type=int, default=30)
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
    parser.add_argument("--init_generator_ckpt", type=str, default=find_default_generator_ckpt())
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()
    train_gan(args)
