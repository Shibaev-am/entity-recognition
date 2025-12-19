import subprocess
import sys
from pathlib import Path

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


@app.command("demo-local")
def demo_local():
    demo_path = PROJECT_ROOT / "apps" / "local_run.py"
    subprocess.run(["streamlit", "run", str(demo_path)], check=True)


@app.command("demo-triton")
def demo_triton():
    demo_path = PROJECT_ROOT / "apps" / "triton_run.py"
    subprocess.run(["streamlit", "run", str(demo_path)], check=True)


@app.command("prepare-triton")
def prepare_triton():
    import shutil

    models_dir = PROJECT_ROOT / "models"
    triton_dir = PROJECT_ROOT / "model_repository" / "bert_ner" / "1"

    onnx_model = models_dir / "model.onnx"
    onnx_data = models_dir / "model.onnx.data"

    if not onnx_model.exists():
        typer.echo(
            "❌ ONNX модель не найдена. Сначала запустите: python -m ner.commands to-onnx"
        )
        raise typer.Exit(1)

    triton_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(onnx_model, triton_dir / "model.onnx")
    typer.echo(f"✅ Скопирован: {onnx_model} -> {triton_dir / 'model.onnx'}")

    if onnx_data.exists():
        shutil.copy(onnx_data, triton_dir / "model.onnx.data")
        typer.echo(f"✅ Скопирован: {onnx_data} -> {triton_dir / 'model.onnx.data'}")

    typer.echo("\n✅ Model repository готов для Triton Server!")
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
