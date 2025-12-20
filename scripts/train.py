import os
import subprocess
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger

from ner.dataset import NERDataModule
from ner.model import BERTNERModel

# Абсолютный путь к директории configs
CONFIG_PATH = str(Path(__file__).parent.parent / "configs")


def get_git_commit_id() -> str:
    """Получить текущий git commit id."""
    try:
        commit_id = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        return commit_id
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


class MetricsPlotCallback(Callback):
    """Callback для сохранения графиков метрик в директорию plots."""

    def __init__(self, plot_dir: str):
        super().__init__()
        self.plot_dir = plot_dir
        os.makedirs(plot_dir, exist_ok=True)

        self.train_losses = []
        self.train_steps = []
        self.train_f1_scores = []
        self.train_epochs = []
        self.val_losses = []
        self.val_f1_scores = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_epochs = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if outputs is not None and "loss" in outputs:
            self.train_losses.append(outputs["loss"].item())
            self.train_steps.append(trainer.global_step)
        elif hasattr(outputs, "item"):
            self.train_losses.append(outputs.item())
            self.train_steps.append(trainer.global_step)

    def _to_python(self, value):
        """Конвертировать тензор или число в Python float."""
        if hasattr(value, "item"):
            return value.item()
        return float(value)

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        if "train_f1" in metrics:
            self.train_f1_scores.append(self._to_python(metrics["train_f1"]))
            self.train_epochs.append(epoch)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        if "val_loss" in metrics:
            self.val_losses.append(self._to_python(metrics["val_loss"]))
        if "val_f1" in metrics:
            self.val_f1_scores.append(self._to_python(metrics["val_f1"]))
        if "val_precision" in metrics:
            self.val_precisions.append(self._to_python(metrics["val_precision"]))
        if "val_recall" in metrics:
            self.val_recalls.append(self._to_python(metrics["val_recall"]))
        self.val_epochs.append(epoch)

    def on_train_end(self, trainer, pl_module):
        self._save_plots()

    def _save_plots(self):
        """Сохранить все графики в директорию plots."""
        # График 1: Training Loss
        if self.train_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(self.train_steps, self.train_losses, label="Train Loss", alpha=0.7)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training Loss over Steps")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.plot_dir, "train_loss.png"), dpi=150)
            plt.close()

        # График 2: Validation Loss
        if self.val_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.val_epochs[: len(self.val_losses)],
                self.val_losses,
                "o-",
                label="Validation Loss",
                color="orange",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Validation Loss over Epochs")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.plot_dir, "val_loss.png"), dpi=150)
            plt.close()

        # График 3: Train F1
        if self.train_f1_scores:
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.train_epochs,
                self.train_f1_scores,
                "o-",
                label="Train F1",
                color="blue",
            )
            plt.xlabel("Epoch")
            plt.ylabel("F1 Score")
            plt.title("Training F1 over Epochs")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.savefig(os.path.join(self.plot_dir, "train_f1.png"), dpi=150)
            plt.close()

        # График 4: Validation F1
        if self.val_f1_scores:
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.val_epochs[: len(self.val_f1_scores)],
                self.val_f1_scores,
                "o-",
                label="Val F1",
                color="green",
            )
            plt.xlabel("Epoch")
            plt.ylabel("F1 Score")
            plt.title("Validation F1 over Epochs")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.savefig(os.path.join(self.plot_dir, "val_f1.png"), dpi=150)
            plt.close()

        # График 5: Validation Recall
        if self.val_recalls:
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.val_epochs[: len(self.val_recalls)],
                self.val_recalls,
                "o-",
                label="Val Recall",
                color="red",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Recall")
            plt.title("Validation Recall over Epochs")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.savefig(os.path.join(self.plot_dir, "val_recall.png"), dpi=150)
            plt.close()

        # График 6: Validation Precision
        if self.val_precisions:
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.val_epochs[: len(self.val_precisions)],
                self.val_precisions,
                "o-",
                label="Val Precision",
                color="purple",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Precision")
            plt.title("Validation Precision over Epochs")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.savefig(os.path.join(self.plot_dir, "val_precision.png"), dpi=150)
            plt.close()

        print(f"Plots saved to: {self.plot_dir}")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def train(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    # Создаём директорию для графиков
    os.makedirs(cfg.paths.plot_dir, exist_ok=True)

    dm = NERDataModule(cfg)
    dm.prepare_data()
    dm.setup()

    model = BERTNERModel(
        model_name=cfg.model.name,
        num_labels=len(dm.tag2idx),
        lr=cfg.model.lr,
        idx2tag=dm.idx2tag,
    )

    # Получаем git commit id
    git_commit_id = get_git_commit_id()

    # Гиперпараметры для логирования
    hyperparams = {
        "model_name": cfg.model.name,
        "learning_rate": cfg.model.lr,
        "max_epochs": cfg.trainer.max_epochs,
        "batch_size": cfg.data.batch_size,
        "max_length": cfg.data.max_length,
        "seed": cfg.seed,
        "precision": cfg.trainer.precision,
        "git_commit_id": git_commit_id,
    }

    # WandB Logger (отключен - требуется авторизация)
    wandb_logger = WandbLogger(
        project=cfg.logger.project, name=cfg.logger.name, log_model="all"
    )
    wandb_logger.experiment.config.update(hyperparams)

    # MLflow Logger
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        log_model=True,
    )
    mlflow_logger.log_hyperparams(hyperparams)

    # Логируем полный конфиг как артефакт в MLflow
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    mlflow_logger.log_hyperparams({"full_config": str(config_dict)})

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.paths.model_save_dir,
        filename="bert-ner-{step:04d}-{val_f1:.2f}",
        save_top_k=1,
        monitor="val_f1",
        mode="max",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    metrics_plot_callback = MetricsPlotCallback(plot_dir=cfg.paths.plot_dir)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        precision=cfg.trainer.precision,
        logger=[mlflow_logger],  # Используем только MLflow (wandb отключен)
        callbacks=[checkpoint_callback, lr_monitor, metrics_plot_callback],
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
    )

    trainer.fit(model, dm)

    torch.save(dm.tag2idx, os.path.join(cfg.paths.model_save_dir, "tag2idx.pt"))

    # Логируем графики как артефакты в MLflow
    import mlflow

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    with mlflow.start_run(run_id=mlflow_logger.run_id):
        for plot_file in os.listdir(cfg.paths.plot_dir):
            if plot_file.endswith(".png"):
                mlflow.log_artifact(
                    os.path.join(cfg.paths.plot_dir, plot_file), artifact_path="plots"
                )
        print(f"Plots logged to MLflow run: {mlflow_logger.run_id}")


if __name__ == "__main__":
    train()
