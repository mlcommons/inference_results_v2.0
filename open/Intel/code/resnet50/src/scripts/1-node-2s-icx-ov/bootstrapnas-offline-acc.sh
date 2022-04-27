#!/bin/bash
BASEDIR=$(dirname "$0")
# BOOTSTRAP_MODEL: A, B, or C
BOOTSTRAP_MODEL=$1

CORE_COUNT=40
CORE_COUNT_DIV_2=$(($CORE_COUNT / 2))
CORE_COUNT_MULT_2=$(($CORE_COUNT * 2))

export DATA_DIR=${BUILD_DIRECTORY}/src/imagenet
export MODEL_DIR=${BUILD_DIRECTORY}/src/models
user_conf=user.conf

# GET MODEL SPECIFIC INFO
case $BOOTSTRAP_MODEL in
    "A")
        BOOTSTRAP_VERSION=bootstrapnasA
        filename=BootstrapNAS_A/BootstrapNAS_A.xml
        bootstrapnastype=bootstrapnas
        ;;

    "B")
        BOOTSTRAP_VERSION=bootstrapnasB
        filename=BootstrapNAS_B/BootstrapNAS_B.xml
        bootstrapnastype=bootstrapnas
        user_conf=user_bnasB.conf
        ;;

    "C")
        BOOTSTRAP_VERSION=bootstrapnasC
        filename=BootstrapNAS_C/BootstrapNAS_C.xml
        bootstrapnastype=bootstrapnas_c
        ;;
esac

export OV_MLPERF_BIN=${BUILD_DIRECTORY}/src/Release/ov_mlperf

${OV_MLPERF_BIN} --scenario Offline \
	--mode Accuracy \
    --mlperf_conf ${BASEDIR}/mlperf.conf \
    --user_conf ${BASEDIR}/$user_conf \
	--model_name $bootstrapnastype \
	--dataset imagenet \
	--data_path $DATA_DIR \
	--model_path $MODEL_DIR/$filename \
	--total_sample_count 50000 \
	--nireq 165 \
	--nstreams $CORE_COUNT_MULT_2 \
	--nthreads $CORE_COUNT_MULT_2 \
	--batch_size 2


