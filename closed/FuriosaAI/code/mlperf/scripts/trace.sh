#!/bin/bash -x

if [ -z $1 ];then
    echo "usage: trace.sh <model>"
    exit 1
fi

if [ -z $BATCH_SIZE ];then
BATCH_SIZE=1
fi

run() {
NAME=$1
NPU_CONFIG=$2
DEVICES=$3
MODEL=$4
WORKERS=$5

echo "========================================================================="
echo "NAME: $NAME"
echo "========================================================================="
MLCOMMONS_USE_CACHED_GRAPH=true \
NPU_COMPLETION_CYCLES=0 \
NPU_GLOBAL_CONFIG_PATH=$NPU_CONFIG \
NPU_DEVNAME=$DEVICES \
NPU_PROFILER_PATH=mlperf_${MODEL}_out/profile_npu.json \
NUX_PROFILER_PATH=mlperf_${MODEL}_out/profile_nux.json \
target/release/runner --model $MODEL --scenario SingleStream --config models/mlperf_small.conf --workers $WORKERS --batch-size $BATCH_SIZE && \
cat mlperf_${MODEL}_out/mlperf_log_summary.txt
}

run "Trace: $1" "warboy-2pe" "npu0pe0-1" "$1" 1
