import os
import subprocess
import sys
from pathlib import Path

import typer
from typing_extensions import Annotated

app = typer.Typer(name="ner", help="BERT NER MLOps Project")

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
def to_tensorrt():
    """Конвертация ONNX модели в TensorRT engine."""
    script_path = PROJECT_ROOT / "scripts" / "to_tensorrt.sh"

    if not script_path.exists():
        typer.echo(f"❌ Скрипт не найден: {script_path}")
        raise typer.Exit(1)

    result = subprocess.run(["bash", str(script_path)], cwd=PROJECT_ROOT)

    if result.returncode != 0:
        typer.echo("❌ Ошибка при конвертации в TensorRT")
        raise typer.Exit(result.returncode)

    typer.echo("Конвертация в TensorRT завершена успешно!")


@app.command("demo-local")
def demo_local():
    demo_path = PROJECT_ROOT / "apps" / "local_run.py"
    subprocess.run(["streamlit", "run", str(demo_path)], check=True)


@app.command("run-triton")
def run_triton(
    device: Annotated[
        str,
        typer.Option(help="Использовать GPU или CPU"),
    ] = "gpu",
    backend: Annotated[
        str,
        typer.Option(help="Тип модели: 'onnx' или 'tensorrt'"),
    ] = "onnx",
):
    """Запуск Triton Inference Server с выбранной моделью."""
    # print(device, backend)
    # exit()
    backend = backend.lower()
    if backend not in ("onnx", "tensorrt"):
        typer.echo(
            f"❌ Неизвестный тип модели: {backend}. Используйте 'onnx' или 'tensorrt'"
        )
        raise typer.Exit(1)

    gpu_mode = "gpu" if device == "gpu" else "no-gpu"
    script_path = PROJECT_ROOT / "scripts" / "run_triton_server.sh"

    if not script_path.exists():
        typer.echo(f"❌ Скрипт не найден: {script_path}")
        raise typer.Exit(1)

    result = subprocess.run(
        ["bash", str(script_path), gpu_mode, backend],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        raise typer.Exit(result.returncode)


@app.command("run-app")
def demo_triton(
    backend: str = typer.Argument(
        "onnx",
        help="Бэкенд для инференса: 'onnx' или 'tensorrt'",
    )
):
    if backend not in ("onnx", "tensorrt"):
        typer.echo(
            f"❌ Неизвестный бэкенд: {backend}. Используйте 'onnx' или 'tensorrt'"
        )
        raise typer.Exit(1)

    # Передаём backend в Streamlit через переменную окружения
    env = os.environ.copy()
    env["TRITON_BACKEND"] = backend

    demo_path = PROJECT_ROOT / "apps" / "triton_run.py"
    typer.echo(f"🚀 Запуск демо с бэкендом: {backend.upper()}")
    subprocess.run(["streamlit", "run", str(demo_path)], env=env, check=True)

    # demo_path = PROJECT_ROOT / "apps" / "triton_run.py"
    # subprocess.run(["streamlit", "run", str(demo_path)], check=True)


def main():
    app()


if __name__ == "__main__":
    main()
