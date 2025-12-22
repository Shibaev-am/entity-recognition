#!/bin/bash

REPO_PATH="$(pwd)/model_repository"
ONNX_PATH="$REPO_PATH/bert_ner_onnx/1/model.onnx"
TRT_DIR="$REPO_PATH/bert_ner_tensorrt/1"
TRT_PLAN="$TRT_DIR/model.plan"

mkdir -p "$TRT_DIR"

docker run --gpus all --rm \
  -v "$REPO_PATH:/models" \
  nvcr.io/nvidia/tritonserver:24.05-py3 \
  /usr/src/tensorrt/bin/trtexec \
  --onnx=/models/bert_ner_onnx/1/model.onnx \
  --saveEngine=/models/bert_ner_tensorrt/1/model.plan \
  --fp16 \
  --minShapes=input_ids:1x1,attention_mask:1x1 \
  --optShapes=input_ids:4x128,attention_mask:4x128 \
  --maxShapes=input_ids:8x128,attention_mask:8x128
