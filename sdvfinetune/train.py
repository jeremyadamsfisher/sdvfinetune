import os
from datetime import timedelta

import hydra
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger

from .config import Config
from .data import VideoDataModule
from .model import SVDLightning
from .utils import run_manager


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def train(cfg: Config):
    """Train a GPT model."""
    with run_manager(cfg.disable_wandb, cfg.load_from) as name:
        dm = VideoDataModule()
        model = SVDLightning.from_config(cfg)

        if cfg.compile:
            model = torch.compile(model)

        if cfg.load_from is None:
            model.init_weights()

        dm.prepare_data()
        dm.setup("fit")

        logger_ = CSVLogger("./csv_logs") if cfg.disable_wandb else WandbLogger()
        callbacks = [LearningRateMonitor(logging_interval="step")]

        if cfg.save_to:
            model_cb = ModelCheckpoint(
                dirpath=os.path.join(cfg.save_to, name),
                filename="{epoch}-{tst_loss:.2f}",
                save_top_k=1,
                mode="min",
                monitor="tst_loss",
                train_time_interval=timedelta(minutes=15),
                save_on_train_epoch_end=True,
            )
            callbacks.append(model_cb)

        trainer = L.Trainer(
            max_epochs=cfg.model_config.n_epochs,
            callbacks=callbacks,
            logger=[logger_],
            accelerator="auto",
            profiler="simple" if cfg.profile else None,
            fast_dev_run=10 if cfg.profile else None,
            precision="bf16-mixed",
            accumulate_grad_batches=cfg.model_config.accumulate_grad_batches,
            default_root_dir=cfg.save_to,
        )

        if cfg.load_from:
            trainer.fit(model, dm, ckpt_path=cfg.load_from)
        else:
            trainer.fit(model, dm)

        return model


if __name__ == "__main__":
    train()
