import os

import hydra
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine_from_onnx(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    max_batch_size: int = 8,
    max_seq_len: int = 128,
):
    """Конвертирует ONNX модель в TensorRT engine."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Парсим ONNX модель
    print(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX Parser Error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    # Настраиваем конфигурацию builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 mode enabled")

    # Настраиваем оптимизационные профили для динамических осей
    profile = builder.create_optimization_profile()

    # input_ids: [batch, sequence]
    profile.set_shape(
        "input_ids",
        min=(1, 1),
        opt=(max_batch_size // 2, max_seq_len // 2),
        max=(max_batch_size, max_seq_len),
    )
    # attention_mask: [batch, sequence]
    profile.set_shape(
        "attention_mask",
        min=(1, 1),
        opt=(max_batch_size // 2, max_seq_len // 2),
        max=(max_batch_size, max_seq_len),
    )

    config.add_optimization_profile(profile)

    # Строим engine
    print("Building TensorRT engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # Сохраняем engine
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"TensorRT engine saved to: {engine_path}")
    return engine_path


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def convert_to_tensorrt(cfg):
    model_dir = cfg.paths.model_save_dir
    onnx_path = os.path.join(model_dir, "model.onnx")
    engine_path = os.path.join(model_dir, "model.engine")

    if not os.path.exists(onnx_path):
        print(f"ONNX model not found at {onnx_path}")
        print("Please run 'ner to-onnx' first to export the model to ONNX format.")
        return

    # Параметры конвертации (можно вынести в конфиг)
    fp16 = getattr(cfg, "tensorrt_fp16", True)
    max_batch_size = getattr(cfg, "tensorrt_max_batch", 8)
    max_seq_len = getattr(cfg, "tensorrt_max_seq_len", 128)

    build_engine_from_onnx(
        onnx_path=onnx_path,
        engine_path=engine_path,
        fp16=fp16,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )


if __name__ == "__main__":
    convert_to_tensorrt()
