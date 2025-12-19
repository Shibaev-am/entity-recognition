import os

import hydra
import numpy as np
import onnxruntime as ort
import torch
from transformers import BertTokenizerFast

SAMPLE_TEXT = "Steve Jobs founded Apple in Cupertino."


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def infer(cfg):
    model_dir = cfg.paths.model_save_dir
    onnx_path = os.path.join(model_dir, "model.onnx")
    tag2idx_path = os.path.join(model_dir, "tag2idx.pt")

    if not os.path.exists(onnx_path):
        print("ONNX model not found. Run scripts/to_onnx.py first.")
        return

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)

    tokenizer = BertTokenizerFast.from_pretrained(cfg.model.name)
    tag2idx = torch.load(tag2idx_path)
    idx2tag = {v: k for k, v in tag2idx.items()}

    inputs = tokenizer(
        SAMPLE_TEXT,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=128,
    )

    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)

    outputs = session.run(
        None, {"input_ids": input_ids, "attention_mask": attention_mask}
    )
    logits = outputs[0]
    preds = np.argmax(logits, axis=2)[0]  # [Seq_len]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    result = []
    for token, pred_idx in zip(tokens, preds):
        if token not in ["[CLS]", "[SEP]", "[PAD]"]:
            tag = idx2tag.get(pred_idx, "O")
            if not token.startswith("##"):
                result.append(f"{token}: {tag}")

    print(f"Text: {SAMPLE_TEXT}")
    print("Entities:", result)


if __name__ == "__main__":
    infer()
