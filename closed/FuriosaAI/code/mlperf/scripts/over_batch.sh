#!/bin/bash -x

if [ -z $1 ] || [ -z $2 ];then
    echo "usage: over_batch.sh <MODEL> <OUTPUT>"
    exit 1
fi

MODEL=$1
OUTPUT=$2

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
target/release/runner --model $NAME --scenario Offline --config models/mlperf_large.conf --workers $WORKERS --batch-size $BATCH && \
grep "Samples per second" mlperf_${MODEL}_out/mlperf_log_summary.txt | awk -v name=$NAME -v batch=$BATCH -v devices=$DEVICES '{print "|"name"|"devices"|"batch"|"$4"|"}' >> $OUTPUT
cat mlperf_${MODEL}_out/mlperf_log_summary.txt
}

for BATCH in 1 2 4 8
do
    run "$MODEL" "npu0pe0-1" "warboy-2pe" 4 $BATCH
done
