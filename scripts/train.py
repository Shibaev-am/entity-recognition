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
    """
    Callback для сохранения графиков метрик в директорию plots.
    Строит графики зависимости метрик от глобального шага (Steps).
    """

    def __init__(self, plot_dir: str):
        super().__init__()
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "train_loss": [],
            "train_f1": [],
            "val_loss": [],
            "val_f1": [],
            "val_precision": [],
            "val_recall": [],
        }

    def _to_python(self, value):
        """Конвертировать тензор или число в Python float."""
        if hasattr(value, "item"):
            return value.item()
        return float(value)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Собираем метрики обучения на каждом шаге."""
        step = trainer.global_step

        if outputs is not None:
            if isinstance(outputs, dict) and "loss" in outputs:
                self.history["train_loss"].append(
                    (step, self._to_python(outputs["loss"]))
                )
            elif hasattr(outputs, "item"):
                self.history["train_loss"].append((step, self._to_python(outputs)))

        metrics = trainer.callback_metrics
        if "train_f1" in metrics:
            self.history["train_f1"].append(
                (step, self._to_python(metrics["train_f1"]))
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Собираем метрики валидации.
        Так как val_check_interval=100, этот метод будет вызываться каждые 100 шагов.
        """
        step = trainer.global_step
        metrics = trainer.callback_metrics

        keys_map = {
            "val_loss": "val_loss",
            "val_f1": "val_f1",
            "val_precision": "val_precision",
            "val_recall": "val_recall",
        }

        for internal_key, log_key in keys_map.items():
            if log_key in metrics:
                self.history[internal_key].append(
                    (step, self._to_python(metrics[log_key]))
                )

    def on_train_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            self._save_plots()

    def _save_plots(self):
        """
        Сохранение графиков в стиле WandB (Dark Mode)
        с УМНЫМ МАСШТАБИРОВАНИЕМ (Robust Scaling), игнорирующим выбросы.
        """
        import numpy as np

        print(f"Saving plots to {self.plot_dir}...")

        plt.style.use("dark_background")

        plots_config = [
            ("train_loss", "Training Loss", "train_loss.png", "#29b5e8"),
            ("val_loss", "Validation Loss", "val_loss.png", "#ff9900"),
            ("val_f1", "Validation F1", "val_f1.png", "#29b5e8"),
            ("val_precision", "Validation Precision", "val_precision.png", "#d16ff5"),
            ("val_recall", "Validation Recall", "val_recall.png", "#ff5050"),
        ]

        for key, title, filename, color in plots_config:
            data = self.history[key]
            if not data:
                continue

            steps, values = zip(*data)
            values_arr = np.array(values)

            if "train_loss" in key.lower():
                fig = plt.figure(figsize=(20, 6), facecolor="#1e1e1e")
            else:
                fig = plt.figure(figsize=(10, 6), facecolor="#1e1e1e")
            ax = plt.gca()
            ax.set_facecolor("#1e1e1e")

            # Рисуем линию
            plt.plot(
                steps,
                values,
                label=title,
                color=color,
                alpha=0.9,
                linewidth=2,
                marker=".",
                markersize=8,
            )

            if "train_loss" in key.lower():
                y_min = 0
                y_max = 0.5
            elif "loss" in key.lower():
                y_min = np.min(values_arr)
                y_max = np.percentile(values_arr, 95)
            else:
                y_min = np.percentile(values_arr, 5)
                y_max = np.max(values_arr)

            y_range = y_max - y_min
            if y_range == 0:
                y_range = 0.1
            padding = y_range * 0.1

            plt.ylim(y_min - padding, y_max + padding)

            from matplotlib.ticker import AutoMinorLocator

            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            plt.grid(
                True,
                which="major",
                color="white",
                linestyle="-",
                linewidth=0.5,
                alpha=0.15,
            )
            plt.grid(
                True,
                which="minor",
                color="white",
                linestyle=":",
                linewidth=0.3,
                alpha=0.05,
            )

            plt.xlabel("Global Step", color="white", fontsize=12)
            plt.ylabel(title, color="white", fontsize=12)
            plt.title(f"{title} over Steps", color="white", fontsize=14, pad=15)
            plt.legend(
                frameon=True, facecolor="#2b2b2b", edgecolor="white", labelcolor="white"
            )

            for spine in ax.spines.values():
                spine.set_color("white")
            ax.tick_params(axis="both", colors="white", which="both")

            save_path = self.plot_dir / filename
            plt.savefig(
                save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
            )
            plt.close()

        plt.style.use("default")
        print("Plots saved successfully.")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def train(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    Path(cfg.paths.plot_dir).mkdir(parents=True, exist_ok=True)

    dm = NERDataModule(cfg)
    dm.prepare_data()
    dm.setup()

    Path(cfg.paths.model_save_dir).mkdir(parents=True, exist_ok=True)
    torch.save(dm.tag2idx, Path(cfg.paths.model_save_dir) / "tag2idx.pt")

    model = BERTNERModel(
        model_name=cfg.model.name,
        num_labels=len(dm.tag2idx),
        lr=cfg.model.lr,
        idx2tag=dm.idx2tag,
    )

    git_commit_id = get_git_commit_id()

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

    wandb_logger = WandbLogger(
        project=cfg.logger.project, name=cfg.logger.name, log_model="all"
    )
    wandb_logger.experiment.config.update(hyperparams)

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        log_model=True,
    )
    mlflow_logger.log_hyperparams(hyperparams)

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    mlflow_logger.log_hyperparams({"full_config": str(config_dict)})

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.paths.model_save_dir,
        filename="checkpoint-step-{step:04d}",
        save_top_k=1,
        monitor="val_f1",
        mode="max",
        auto_insert_metric_name=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    metrics_plot_callback = MetricsPlotCallback(plot_dir=cfg.paths.plot_dir)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        precision=cfg.trainer.precision,
        logger=[mlflow_logger, wandb_logger],
        callbacks=[checkpoint_callback, lr_monitor, metrics_plot_callback],
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
    )

    trainer.fit(model, dm)
    
    torch.save(dm.tag2idx, Path(cfg.paths.model_save_dir) / "tag2idx.pt")

    if trainer.is_global_zero:
        plot_dir = Path(cfg.paths.plot_dir)
        print(f"Logging plots to MLflow run: {mlflow_logger.run_id}")
        for plot_file in plot_dir.iterdir():
            if plot_file.suffix == ".png":
                mlflow_logger.experiment.log_artifact(
                    run_id=mlflow_logger.run_id,
                    local_path=str(plot_file),
                    artifact_path="plots",
                )


if __name__ == "__main__":
    train()
