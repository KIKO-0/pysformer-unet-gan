from root import ROOT_DIR
import argparse

import lightning.pytorch as pl
import torch
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.tuner import Tuner

from models import unet_precip_regression_lightning as unet_regr


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


def build_model(hparams):
    if hparams.model in {"UNetDS_CoordAtt", "PhysFormerUNet"}:
        return unet_regr.PhysFormerUNet(hparams=hparams)
    if hparams.model == "UNetDSAttention":
        return unet_regr.UNetDSAttention(hparams=hparams)
    if hparams.model == "UNetAttention":
        return unet_regr.UNetAttention(hparams=hparams)
    if hparams.model == "UNet":
        return unet_regr.UNet(hparams=hparams)
    if hparams.model == "UNetDS":
        return unet_regr.UNetDS(hparams=hparams)
    raise NotImplementedError(f"Model '{hparams.model}' not implemented")


def train_regression(hparams, find_batch_size_automatically: bool = False):
    net = build_model(hparams)
    default_save_path = ROOT_DIR / "lightning" / "precip_regression"

    checkpoint_callback = ModelCheckpoint(
        dirpath=default_save_path / net.__class__.__name__,
        filename=net.__class__.__name__ + "_{epoch}-{val_loss:.6f}",
        save_top_k=-1,
        verbose=False,
        monitor="val_loss",
        mode="min",
    )
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=default_save_path / net.__class__.__name__,
        filename=net.__class__.__name__ + "_last",
        save_top_k=1,
        verbose=False,
    )

    lr_monitor = LearningRateMonitor()
    tb_logger = loggers.TensorBoardLogger(save_dir=default_save_path, name=net.__class__.__name__)
    earlystopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=hparams.es_patience,
    )

    accelerator, devices = resolve_training_device()
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        fast_dev_run=hparams.fast_dev_run,
        max_epochs=hparams.epochs,
        default_root_dir=default_save_path,
        logger=tb_logger,
        callbacks=[checkpoint_callback, last_checkpoint_callback, earlystopping_callback, lr_monitor],
        val_check_interval=hparams.val_check_interval,
    )

    if find_batch_size_automatically:
        tuner = Tuner(trainer)
        tuner.scale_batch_size(net, mode="binsearch")

    trainer.fit(model=net, ckpt_path=hparams.resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--fast_dev_run", type=bool, default=False)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--val_check_interval", type=float, default=1.0)

    parser.add_argument("--n_channels", type=int, default=12)
    parser.add_argument("--n_classes", type=int, default=1)
    parser.add_argument("--bilinear", type=bool, default=True)
    parser.add_argument("--kernels_per_layer", type=int, default=2)
    parser.add_argument("--reduction_ratio", type=int, default=16)
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=1)
    parser.add_argument("--transformer_dropout", type=float, default=0.1)

    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--es_patience", type=int, default=15)
    parser.add_argument("--physics_threshold", type=float, default=0.3)
    parser.add_argument("--physics_rain_weight", type=float, default=4.0)
    parser.add_argument("--physics_edge_weight", type=float, default=0.1)

    parser.add_argument("--dataset_folder", type=str, default="")
    parser.add_argument("--use_oversampled_dataset", type=bool, default=True)
    parser.add_argument("--num_input_images", type=int, default=12)
    parser.add_argument("--num_output_images", type=int, default=1)
    parser.add_argument("--valid_size", type=float, default=0.1)

    args = parser.parse_args()

    target_models = ["PhysFormerUNet"]
    for model_name in target_models:
        args.model = model_name
        print("==================================================")
        print(f" Start training INNOVATION MODEL: {model_name}")
        print("==================================================")
        train_regression(args, find_batch_size_automatically=False)
