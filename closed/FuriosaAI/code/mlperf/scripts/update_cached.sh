#! /bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export RUST_BACKTRACE=1
export MLCOMMONS_USE_CACHED_GRAPH=true

compiler=target/release/compile_model

resnet50=models/resnet50_int8.onnx
ssd_small=models/ssd_mobilenet_int8.onnx
ssd_large=models/ssd_resnet34_int8.onnx

echo "== resnet50 for SingleStream =="
NPU_GLOBAL_CONFIG_PATH=warboy-2pe "${compiler}" --model "${resnet50}" --batch-size 1 --remove-unlower &
sleep 3

echo "== Compile resnet50 for Offline =="
NPU_GLOBAL_CONFIG_PATH=warboy "${compiler}" --model "${resnet50}" --batch-size 8 --remove-unlower &
sleep 3

echo "== ssd_small for SingleStream =="
NPU_GLOBAL_CONFIG_PATH=warboy-2pe "${compiler}" --model "${ssd_small}" --batch-size 1 --remove-unlower &
sleep 3

echo "== ssd_small for Offline =="
NPU_GLOBAL_CONFIG_PATH=warboy "${compiler}" --model "${ssd_small}" --batch-size 8 --remove-unlower &
sleep 3

echo "== ssd_large for SingleStream and Offline =="
NPU_GLOBAL_CONFIG_PATH=warboy-2pe "${compiler}" --model "${ssd_large}" --batch-size 1 --remove-unlower &
