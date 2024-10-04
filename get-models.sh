#!/bin/bash
echo "Downloading models from Hugging Face"
echo ""

echo "Downloading files for Phi-3-mini-4k-instruct-onnx-web..."
curl -L https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx-web/resolve/main/config.json -o models/microsoft/Phi-3-mini-4k-instruct-onnx-web/config.json
curl -L https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx-web/resolve/main/onnx/model_q4f16.onnx -o models/microsoft/Phi-3-mini-4k-instruct-onnx-web/onnx/model_q4f16.onnx
curl -L https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx-web/resolve/main/onnx/model_q4f16.onnx_data -o models/microsoft/Phi-3-mini-4k-instruct-onnx-web/onnx/model_q4f16.onnx_data
curl -L https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx-web/resolve/main/tokenizer.json -o models/microsoft/Phi-3-mini-4k-instruct-onnx-web/tokenizer.json
curl -L https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx-web/resolve/main/tokenizer_config.json -o models/microsoft/Phi-3-mini-4k-instruct-onnx-web/tokenizer_config.json
curl -L https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx-web/resolve/main/tokenizer.model -o models/microsoft/Phi-3-mini-4k-instruct-onnx-web/tokenizer.model
