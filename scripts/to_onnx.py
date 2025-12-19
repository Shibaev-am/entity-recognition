import os

import hydra
import torch

from ner.model import BERTNERModel


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def convert_to_onnx(cfg):
    model_dir = cfg.paths.model_save_dir
    checkpoints = [f for f in os.listdir(model_dir) if f.endswith(".ckpt")]
    if not checkpoints:
        print("No checkpoints found!")
        return

    ckpt_path = os.path.join(model_dir, checkpoints[0])
    print(f"Loading checkpoint: {ckpt_path}")

    # Загрузка весов
    # ВАЖНО: Нам нужно знать количество классов. В реальном коде нужно загрузить tag2idx.pt
    tag2idx = torch.load(os.path.join(model_dir, "tag2idx.pt"), map_location="cpu")
    model = BERTNERModel.load_from_checkpoint(
        ckpt_path,
        model_name=cfg.model.name,
        num_labels=len(tag2idx),
        lr=cfg.model.lr,
        idx2tag={v: k for k, v in tag2idx.items()},
        map_location="cpu",
    )
    model.to("cpu")
    model.eval()

    dummy_input = torch.randint(0, 1000, (1, 128), device="cpu")  # [Batch, Seq]
    dummy_mask = torch.ones((1, 128), dtype=torch.long, device="cpu")

    output_path = os.path.join(model_dir, "model.onnx")

    print(
        f"Exporting with params: {sum(p.numel() for p in model.model.parameters())} parameters"
    )
    torch.onnx.export(
        model.model,
        (dummy_input, dummy_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=18,
        export_params=True,
    )
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    convert_to_onnx()
