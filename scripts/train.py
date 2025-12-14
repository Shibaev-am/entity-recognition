import os

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.dataset import NERDataModule
from src.model import BERTNERModel


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    dm = NERDataModule(cfg)
    dm.prepare_data()
    dm.setup()

    model = BERTNERModel(
        model_name=cfg.model.name,
        num_labels=len(dm.tag2idx),
        lr=cfg.model.lr,
        idx2tag=dm.idx2tag,
    )

    wandb_logger = WandbLogger(
        project=cfg.logger.project, name=cfg.logger.name, log_model="all"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.paths.model_save_dir,
        filename="bert-ner-{step:04d}-{val_f1:.2f}",
        save_top_k=1,
        monitor="val_f1",
        mode="max",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        precision=cfg.trainer.precision,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=cfg.trainer.log_every_n_steps,  # 200
        val_check_interval=cfg.trainer.val_check_interval,  # 500
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,  # null
    )

    trainer.fit(model, dm)

    torch.save(dm.tag2idx, os.path.join(cfg.paths.model_save_dir, "tag2idx.pt"))


if __name__ == "__main__":
    train()
