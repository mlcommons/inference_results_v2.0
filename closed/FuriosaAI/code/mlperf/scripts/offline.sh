#!/bin/bash -x

if [ -z $1 ];then
    echo "usage: offline.sh <OUTPUT>"
    exit 1
fi

OUTPUT=$1

run() {
NAME=$1
DEVICES=$2
NPU_CONFIG=$3
WORKERS=$4
BATCH=$5

echo "========================================================================="
echo "NAME: $NAME"
echo "========================================================================="
MLCOMMONS_USE_CACHED_GRAPH=true \
NPU_COMPLETION_CYCLES=0 \
NPU_GLOBAL_CONFIG_PATH=$NPU_CONFIG \
NPU_DEVNAME=$DEVICES \
nice -n 0 -- target/release/runner --model $NAME --scenario Offline --config models/mlperf_large.conf --workers $WORKERS --batch-size $BATCH && \
grep "Samples per second" mlperf_${NAME}_out/mlperf_log_summary.txt | awk -v name=$NAME -v devices=$DEVICES '{print "|"name"|"devices"|"$4"|"}' >> $OUTPUT
cat mlperf_${NAME}_out/mlperf_log_summary.txt
}

run "resnet50" "npu0pe0,npu0pe1" "warboy" 4 8
run "ssd_small" "npu0pe0,npu0pe1" "warboy" 8 8
run "ssd_large" "npu0pe0-1" "warboy-2pe" 4 1
