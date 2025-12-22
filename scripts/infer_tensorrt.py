import os
from pathlib import Path

import hydra
import numpy as np

# Абсолютный путь к директории configs
CONFIG_PATH = str(Path(__file__).parent.parent / "configs")

# pycuda нужен для работы с GPU памятью в TensorRT
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
import torch
from transformers import BertTokenizerFast

SAMPLE_TEXT = "Steve Jobs founded Apple in Cupertino."

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TensorRTInference:
    """Класс для инференса TensorRT модели."""

    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.stream = None
        self.bindings = []
        self.inputs = {}
        self.outputs = {}

        self._load_engine()
        self._allocate_buffers()

    def _load_engine(self):
        """Загрузка TensorRT engine."""
        runtime = trt.Runtime(TRT_LOGGER)
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

    def _allocate_buffers(self):
        """Выделение буферов для входов и выходов."""
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)

            # Для динамических осей используем максимальные значения
            shape = [s if s > 0 else 8 for s in shape]  # batch=8 как fallback
            size = int(np.prod(shape))

            # Выделяем память на GPU
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)

            binding = {
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "device": device_mem,
            }

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = binding
            else:
                self.outputs[name] = binding

    def infer(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Выполнение инференса."""
        batch_size, seq_len = input_ids.shape

        # Устанавливаем динамические размеры
        self.context.set_input_shape("input_ids", (batch_size, seq_len))
        self.context.set_input_shape("attention_mask", (batch_size, seq_len))

        # Копируем входные данные на GPU
        input_ids_flat = np.ascontiguousarray(input_ids.astype(np.int64))
        attention_mask_flat = np.ascontiguousarray(attention_mask.astype(np.int64))

        cuda.memcpy_htod_async(
            self.inputs["input_ids"]["device"], input_ids_flat, self.stream
        )
        cuda.memcpy_htod_async(
            self.inputs["attention_mask"]["device"], attention_mask_flat, self.stream
        )

        # Устанавливаем адреса тензоров
        self.context.set_tensor_address(
            "input_ids", int(self.inputs["input_ids"]["device"])
        )
        self.context.set_tensor_address(
            "attention_mask", int(self.inputs["attention_mask"]["device"])
        )

        # Выделяем память для выхода с правильным размером
        output_name = list(self.outputs.keys())[0]
        output_shape = self.context.get_tensor_shape(output_name)
        output_size = int(np.prod(output_shape))
        output_dtype = self.outputs[output_name]["dtype"]

        # Перевыделяем буфер если нужно
        output_device = cuda.mem_alloc(output_size * np.dtype(output_dtype).itemsize)
        self.context.set_tensor_address(output_name, int(output_device))

        # Выполняем инференс
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Копируем результат обратно на CPU
        output_host = np.empty(output_shape, dtype=output_dtype)
        cuda.memcpy_dtoh_async(output_host, output_device, self.stream)

        self.stream.synchronize()

        return output_host

    def __del__(self):
        """Освобождение ресурсов."""
        if self.stream:
            self.stream.synchronize()


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def infer_tensorrt(cfg):
    model_dir = cfg.paths.model_save_dir
    engine_path = os.path.join(model_dir, "model.engine")
    tag2idx_path = os.path.join(model_dir, "tag2idx.pt")

    if not os.path.exists(engine_path):
        print(f"TensorRT engine not found at {engine_path}")
        print("Please run 'ner to-tensorrt' first to convert the model.")
        return

    # Загружаем TensorRT engine
    print(f"Loading TensorRT engine: {engine_path}")
    trt_model = TensorRTInference(engine_path)

    # Загружаем токенизатор и маппинг тегов
    tokenizer = BertTokenizerFast.from_pretrained(cfg.model.name)
    tag2idx = torch.load(tag2idx_path, map_location="cpu")
    idx2tag = {v: k for k, v in tag2idx.items()}

    # Токенизация входного текста
    inputs = tokenizer(
        SAMPLE_TEXT,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=128,
    )

    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)

    # Инференс
    logits = trt_model.infer(input_ids, attention_mask)
    preds = np.argmax(logits, axis=2)[0]  # [Seq_len]

    # Декодирование результатов
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
    infer_tensorrt()
