#! /bin/bash

# Uncomment next line to check script without actual running
# export DRY_RUN=true

target_npus="1,3,4"
run_name="nightly_example_$(date +'%Y%m%d')"
export DATASET_ROOT="/home/ubuntu/preprocessed"
export MLFLOW_RUN_NAME="${run_name}"

# check device accuracy
DISABLE_PERF=true REPEAT_ACCURACY_ON_NPUS=true \
./scripts/param_sweep.sh "${target_npus}" "outputs/${run_name}_accuracy"

# performance
DISABLE_ACCURACY=true RUN_ALL=false \
RUN_RESNET50=true REPEAT_SINGLE_STREAM_ON_NPUS=true REPEAT_OFFLINE_ON_NPUS=true SCALE_OFFLINE_ON_NPUS=true \
./scripts/param_sweep.sh "${target_npus}" "outputs/${run_name}"

DISABLE_ACCURACY=true RUN_ALL=false \
RUN_SSD_SMALL_SINGLE_STREAM=true REPEAT_SINGLE_STREAM_ON_NPUS=true REPEAT_ON_TASKSETS=true \
./scripts/param_sweep.sh "${target_npus}" "outputs/${run_name}"

DISABLE_ACCURACY=true RUN_ALL=false \
RUN_SSD_SMALL_OFFLINE=true REPEAT_OFFLINE_ON_NPUS=true REPEAT_ON_TASKSETS=true \
./scripts/param_sweep.sh "${target_npus}" "outputs/${run_name}"

MLPERF_CONFIG=scripts/mlperf_large2.conf DISABLE_ACCURACY=true RUN_ALL=false \
RUN_SSD_SMALL_OFFLINE=true REPEAT_OFFLINE_ON_TWO_NPUS=true REPEAT_ON_TASKSETS=true \
./scripts/param_sweep.sh "${target_npus}" "outputs/${run_name}"

MLPERF_CONFIG=scripts/mlperf_large3.conf DISABLE_ACCURACY=true RUN_ALL=false \
RUN_SSD_SMALL_OFFLINE=true REPEAT_OFFLINE_ON_THREE_NPUS=true REPEAT_ON_TASKSETS=true \
./scripts/param_sweep.sh "${target_npus}" "outputs/${run_name}"

DISABLE_ACCURACY=true RUN_ALL=false \
RUN_SSD_LARGE=true REPEAT_SINGLE_STREAM_ON_NPUS=true REPEAT_OFFLINE_ON_NPUS=true SCALE_OFFLINE_ON_NPUS=true \
./scripts/param_sweep.sh "${target_npus}" "outputs/${run_name}"
