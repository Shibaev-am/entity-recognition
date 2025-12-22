import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="ner",
    help="BERT NER MLOps Project",
    add_completion=False,
)

PROJECT_ROOT = Path(__file__).parent.parent


@app.command()
def train(
    overrides: list[str] = typer.Argument(
        None, help="Hydra overrides, например: trainer.max_epochs=5"
    ),
):
    from scripts.train import train as train_fn

    sys.argv = ["train"]
    if overrides:
        sys.argv.extend(overrides)
    train_fn()


@app.command("to-onnx")
def to_onnx(
    overrides: list[str] = typer.Argument(
        None, help="Hydra overrides, например: paths.model_save_dir=./other_models"
    ),
):
    from scripts.to_onnx import convert_to_onnx

    sys.argv = ["to_onnx"]
    if overrides:
        sys.argv.extend(overrides)
    convert_to_onnx()


@app.command()
def infer(
    overrides: list[str] = typer.Argument(None, help="Hydra overrides"),
):
    from scripts.infer import infer as infer_fn

    sys.argv = ["infer"]
    if overrides:
        sys.argv.extend(overrides)
    infer_fn()


@app.command("to-tensorrt")
def to_tensorrt(
    overrides: list[str] = typer.Argument(
        None, help="Hydra overrides, например: paths.model_save_dir=./other_models"
    ),
):
    """Конвертация ONNX модели в TensorRT engine."""
    from scripts.to_tensorrt import convert_to_tensorrt

    sys.argv = ["to_tensorrt"]
    if overrides:
        sys.argv.extend(overrides)
    convert_to_tensorrt()


@app.command("infer-tensorrt")
def infer_tensorrt(
    overrides: list[str] = typer.Argument(None, help="Hydra overrides"),
):
    """Инференс модели с использованием TensorRT."""
    from scripts.infer_tensorrt import infer_tensorrt as infer_trt_fn

    sys.argv = ["infer_tensorrt"]
    if overrides:
        sys.argv.extend(overrides)
    infer_trt_fn()


@app.command("demo-local")
def demo_local():
    demo_path = PROJECT_ROOT / "apps" / "local_run.py"
    subprocess.run(["streamlit", "run", str(demo_path)], check=True)


@app.command("demo-triton")
def demo_triton():
    demo_path = PROJECT_ROOT / "apps" / "triton_run.py"
    subprocess.run(["streamlit", "run", str(demo_path)], check=True)


@app.command("prepare-triton")
def prepare_triton(
    backend: str = typer.Argument(
        "onnx",  # Значение по умолчанию, если ничего не передать
        help="Бэкенд для инференса: 'onnx' или 'tensorrt'",
    ),
):
    """Подготовка model_repository для Triton Server (ONNX или TensorRT)."""
    import shutil

    models_dir = PROJECT_ROOT / "models"
    triton_dir = PROJECT_ROOT / "model_repository" / "bert_ner" / "1"
    config_dir = PROJECT_ROOT / "model_repository" / "bert_ner"

    backend = backend.lower()
    if backend not in ("onnx", "tensorrt"):
        typer.echo(
            f"❌ Неизвестный бэкенд: {backend}. Используйте 'onnx' или 'tensorrt'"
        )
        raise typer.Exit(1)

    triton_dir.mkdir(parents=True, exist_ok=True)

    if backend == "onnx":
        onnx_model = models_dir / "model.onnx"
        onnx_data = models_dir / "model.onnx.data"

        if not onnx_model.exists():
            typer.echo(
                "❌ ONNX модель не найдена. Сначала запустите: python -m ner.commands to-onnx"
            )
            raise typer.Exit(1)

        # Удаляем старые файлы если есть
        for old_file in triton_dir.glob("*"):
            old_file.unlink()

        shutil.copy(onnx_model, triton_dir / "model.onnx")
        typer.echo(f"✅ Скопирован: {onnx_model} -> {triton_dir / 'model.onnx'}")

        if onnx_data.exists():
            shutil.copy(onnx_data, triton_dir / "model.onnx.data")
            typer.echo(f"✅ Скопирован: {onnx_data} -> {triton_dir / 'model.onnx.data'}")

        # Используем конфиг для ONNX
        shutil.copy(config_dir / "config.pbtxt", config_dir / "config.pbtxt.bak")
        typer.echo("✅ Используется конфигурация для ONNX бэкенда")

    else:  # tensorrt
        engine_model = models_dir / "model.engine"

        if not engine_model.exists():
            typer.echo(
                "❌ TensorRT engine не найден. Сначала запустите:\n"
                "   python -m ner.commands to-onnx\n"
                "   python -m ner.commands to-tensorrt"
            )
            raise typer.Exit(1)

        # Удаляем старые файлы если есть
        for old_file in triton_dir.glob("*"):
            old_file.unlink()

        shutil.copy(engine_model, triton_dir / "model.plan")
        typer.echo(f"✅ Скопирован: {engine_model} -> {triton_dir / 'model.plan'}")

        # Используем конфиг для TensorRT
        tensorrt_config = config_dir / "config_tensorrt.pbtxt"
        if tensorrt_config.exists():
            shutil.copy(tensorrt_config, config_dir / "config.pbtxt")
            typer.echo("✅ Используется конфигурация для TensorRT бэкенда")
        else:
            typer.echo(
                "⚠️ config_tensorrt.pbtxt не найден, используется текущий config.pbtxt"
            )

    typer.echo(
        f"\n✅ Model repository готов для Triton Server! (бэкенд: {backend.upper()})"
    )
    typer.echo("\nЗапустите Triton командой:")
    typer.echo(
        "docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 "
        "-v $(pwd)/model_repository:/models "
        "nvcr.io/nvidia/tritonserver:24.05-py3 tritonserver --model-repository=/models"
    )


def main():
    app()


if __name__ == "__main__":
    main()
