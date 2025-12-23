#!/bin/bash

set -e

# Параметры по умолчанию
GPU_MODE="${1:-gpu}"
MODEL_TYPE="${2:-onnx}"

# Валидация параметров
if [[ "$GPU_MODE" != "gpu" && "$GPU_MODE" != "no-gpu" ]]; then
    echo "❌ Ошибка: первый параметр должен быть 'gpu' или 'no-gpu'"
    echo "Использование: $0 [gpu|no-gpu] [onnx|tensorrt]"
    exit 1
fi

if [[ "$MODEL_TYPE" != "onnx" && "$MODEL_TYPE" != "tensorrt" ]]; then
    echo "❌ Ошибка: второй параметр должен быть 'onnx' или 'tensorrt'"
    echo "Использование: $0 [gpu|no-gpu] [onnx|tensorrt]"
    exit 1
fi

# TensorRT требует GPU
if [[ "$MODEL_TYPE" == "tensorrt" && "$GPU_MODE" == "no-gpu" ]]; then
    echo "❌ Ошибка: TensorRT модель требует GPU"
    exit 1
fi

# Определяем имя модели
if [[ "$MODEL_TYPE" == "onnx" ]]; then
    MODEL_NAME="bert_ner_onnx"
else
    MODEL_NAME="bert_ner_tensorrt"
fi

# Путь к репозиторию моделей
REPO_PATH="$(pwd)/model_repository"

# Проверяем существование директории модели
if [[ ! -d "$REPO_PATH/$MODEL_NAME" ]]; then
    echo "❌ Ошибка: директория модели не найдена: $REPO_PATH/$MODEL_NAME"
    exit 1
fi

# Формируем команду docker
DOCKER_CMD="docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002"

# Добавляем GPU флаг если нужно
if [[ "$GPU_MODE" == "gpu" ]]; then
    DOCKER_CMD="$DOCKER_CMD --gpus all"
fi

# Добавляем монтирование и параметры Triton
DOCKER_CMD="$DOCKER_CMD -v $REPO_PATH:/models"
DOCKER_CMD="$DOCKER_CMD nvcr.io/nvidia/tritonserver:24.05-py3"
DOCKER_CMD="$DOCKER_CMD tritonserver --model-repository=/models"
DOCKER_CMD="$DOCKER_CMD --model-control-mode=explicit"
DOCKER_CMD="$DOCKER_CMD --load-model=$MODEL_NAME"

echo "🚀 Запуск Triton Inference Server"
echo "   GPU: $GPU_MODE"
echo "   Модель: $MODEL_NAME"
echo ""
echo "Выполняется команда:"
echo "$DOCKER_CMD"
echo ""

# Запуск
exec $DOCKER_CMD
