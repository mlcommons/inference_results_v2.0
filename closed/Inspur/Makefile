# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SHELL := /bin/bash
MAKEFILE_NAME := $(lastword $(MAKEFILE_LIST))

ARCH := $(shell uname -p)
UNAME := $(shell whoami)
UID := $(shell id -u `whoami`)
HOSTNAME := $(shell hostname)
GROUPNAME := $(shell id -gn `whoami`)
GROUPID := $(shell id -g `whoami`)
TIMESTAMP := $(shell date +'%Y.%m.%d-%H.%M.%S')
ifndef HOST_HOSTNAME
    HOST_HOSTNAME := $(HOSTNAME)
endif
HOST_HOSTNAME_SHORT := $(firstword $(subst ., , $(HOST_HOSTNAME)))
HOSTNAME_SHORT := $(firstword $(subst ., , $(HOSTNAME)))
PYTHON3_CMD := $(shell if ! python3 --version | grep 3.8 &> /dev/null; then echo python3.8; else echo python3; fi)

USE_NIGHTLY ?= 1
USE_CPU ?= 0
USE_INFERENTIA ?= 0
MINIMAL_REGRESSION_PRESUBMIT ?= 0

SUBMITTER ?= Inspur

# Arch and board variables.
SYSTEM_NAME ?= $(shell $(PYTHON3_CMD) scripts/get_system_id.py 2> /dev/null)
TARGET_X86_64 := 0
TARGET_AARCH64 := 0

IS_SOC := 0
SOC_SM := 0
ifeq ($(ARCH), x86_64)
    TARGET_X86_64 = 1
endif
ifeq ($(ARCH), aarch64)
    TARGET_AARCH64 = 1
    # Find occurrence of Xavier in the system name
    ifneq (,$(findstring Xavier,$(SYSTEM_NAME)))
    	IS_SOC = 1
    	SOC_SM = 72
    endif
    ifneq (,$(findstring Orin,$(SYSTEM_NAME)))
    	IS_SOC = 1
    	SOC_SM = 87
    endif
endif


# Docker is supported for all non-xavier x86/aarch64 system.
SUPPORT_DOCKER := 0
ifeq ($(ARCH), $(filter $(ARCH), x86_64 aarch64))
ifneq ($(IS_SOC), 1)
    SUPPORT_DOCKER = 1
endif
endif

# Conditional Docker flags
ifndef DOCKER_DETACH
    DOCKER_DETACH := 0
endif
ifndef DOCKER_TAG
    DOCKER_TAG := $(UNAME)-$(ARCH)
endif
DOCKER_BUILDKIT ?= 1

DOCKER_NAME := mlperf-inference-$(DOCKER_TAG)

PROJECT_ROOT := $(shell pwd)
BUILD_DIR    := $(PROJECT_ROOT)/build

HOST_VOL ?= ${PWD}
CONTAINER_VOL ?= /work
NO_DOCKER_PULL ?= 0
NO_BUILD ?= 0

# nsys and nvprof locked clock frequency
GPUCLK?=1000

# Set the include directory for Loadgen header files
INFERENCE_DIR = $(BUILD_DIR)/inference
INFERENCE_URL = https://github.com/mlcommons/inference.git
LOADGEN_INCLUDE_DIR ?= $(INFERENCE_DIR)/loadgen
LOADGEN_LIB_DIR ?= $(LOADGEN_INCLUDE_DIR)/build
INFERENCE_HASH = 701847516c3a857e15a61bf4276eeaadd7416f36

# Set the power-dev directory
POWER_DEV_DIR = $(BUILD_DIR)/power-dev
POWER_DEV_URL = https://github.com/mlcommons/power-dev.git
POWER_DEV_HASH = f7b3b51166b614fab0d7b913822943cc3b414dba
POWER_CLIENT_SCRIPT = $(BUILD_DIR)/power-dev/ptd_client_server/client.py
POWER_NTP_URL = ntp.aliyun.com
POWER_SERVER_IP = 10.49.42.32

POWER_SERVER_USERNAME = mlperf
POWER_SERVER_PASSWORD = R5t6y7u/
POWER_SERVER_USER_IP = $(POWER_SERVER_USERNAME)@$(POWER_SERVER_IP)
POWER_SERVER_CONFIG = power/server-inspur.cfg
POWER_SERVER_SCRIPT_DIR = C:/power-dev/ptd_client_server
POWER_SERVER_LOG_DIR = C:/ptd-logs
POWER_SSH = sshpass -p $(POWER_SERVER_PASSWORD) ssh -o "StrictHostKeyChecking no"
POWER_SCP = sshpass -p $(POWER_SERVER_PASSWORD) scp -o "StrictHostKeyChecking no"

# Set log directories for power logs
POWER_LOGS_TEMP_DIR = $(BUILD_DIR)/power_logs_temp

# Set the Triton directory
TRITON_DIR = $(BUILD_DIR)/triton-inference-server
TRITON_OUT_DIR = $(TRITON_DIR)/out
TRITON_URL = https://github.com/triton-inference-server/server
TRITON_HASH = ec1881d491cc2d2bd89ad9383724118e7121280c
TRITON_COMMON_HASH = c009eebc38aa86913dc933e9d075262d8b1a38b2
TRITON_CORE_HASH = dc912a0d60d16155f0e68e53d9f198fddf4d3922
TRITON_BACKEND_HASH = 50b337b7528c3fec1195c31fa299218c3bffd8cd
TRITON_THIRDPARTY_HASH = 772eb32437e3f34a7d2d007ddb04c5e47328c174
TRITON_TENSORRT_HASH = pull/21/head
TRITON_PYTHON_BE_HASH = pull/106/head

# Set this to 0 once repo is frozen
CHECK_TRITON_VERSION=0
BYPASS_TRITON_WARNING=0

# Set Environment variables to extracted contents
export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):/usr/local/cuda/lib64:/usr/lib/$(ARCH)-linux-gnu:$(LOADGEN_LIB_DIR)
export LIBRARY_PATH := /usr/local/cuda/lib64:/usr/lib/$(ARCH)-linux-gnu:$(LOADGEN_LIB_DIR):$(LIBRARY_PATH)
export PATH := /usr/local/cuda/bin:$(PATH)
export CPATH := /usr/local/cuda/include:/usr/include/$(ARCH)-linux-gnu:/usr/include/$(ARCH)-linux-gnu/cub:$(CPATH)
export CUDA_PATH := /usr/local/cuda
export CCACHE_DISABLE=1
export NUMBA_CACHE_DIR=$(BUILD_DIR)/cache

# Set CUDA_DEVICE_MAX_CONNECTIONS to increase multi-stream performance.
export CUDA_DEVICE_MAX_CONNECTIONS := 32

# Please run `export MLPERF_SCRATCH_PATH=<path>` to set your scratch space path.
# The below paths are for internal use only.
ifneq ($(wildcard /home/mlperf_inference_data),)
    MLPERF_SCRATCH_PATH ?= /home/mlperf_inference_data
endif
ifneq ($(wildcard /home/scratch.mlperf_inference),)
    DOCKER_MOUNTS += -v /home/scratch.mlperf_inference:/home/scratch.mlperf_inference
endif
ifneq ($(wildcard /home/scratch.svc_compute_arch),)
    DOCKER_MOUNTS += -v /home/scratch.svc_compute_arch:/home/scratch.svc_compute_arch
endif
ifneq ($(wildcard /home/scratch.computelab/sudo),)
    DOCKER_MOUNTS += -v /home/scratch.computelab/sudo:/home/scratch.computelab/sudo
endif
ifneq ($(wildcard /home/scratch.dlsim),)
    DOCKER_MOUNTS += -v /home/scratch.dlsim:/home/scratch.dlsim
endif
ifneq ($(wildcard $(PROJECT_ROOT)/../../regression),)
    DOCKER_MOUNTS += -v $(PROJECT_ROOT)/../../regression:/regression
endif
ifdef MLPERF_SCRATCH_PATH
    ifneq ($(wildcard $(MLPERF_SCRATCH_PATH)),)
        DOCKER_MOUNTS += -v $(MLPERF_SCRATCH_PATH):$(MLPERF_SCRATCH_PATH)
    else
        $(error Path set in MLPERF_SCRATCH_PATH does not exist!)
    endif
endif

# DATA_DIR is the actual location of data in the user-specified MLPERF_SCRATCH_PATH location.
# On the other hand, DATA_DIR_LINK is the location which our scripts assume the data to be located in. In the
# "link_dirs" target, we create a symbolic from DATA_DIR_LINK to DATA_DIR. The same applies to PREPROCESSED_DATA_DIR and
# MODEL_DIR as well.
DATA_DIR_LINK := $(BUILD_DIR)/data
PREPROCESSED_DATA_DIR_LINK := $(BUILD_DIR)/preprocessed_data
MODEL_DIR_LINK := $(BUILD_DIR)/models
DATA_DIR ?= $(MLPERF_SCRATCH_PATH)/data
PREPROCESSED_DATA_DIR ?= $(MLPERF_SCRATCH_PATH)/preprocessed_data
MODEL_DIR ?= $(MLPERF_SCRATCH_PATH)/models
export DATA_DIR
export PREPROCESSED_DATA_DIR
export MODEL_DIR

# Specify default dir for harness output logs.
ifndef LOG_DIR
    export LOG_DIR := $(BUILD_DIR)/logs/$(TIMESTAMP)
    export POWER_LOGS_DIR := $(BUILD_DIR)/power_logs/$(TIMESTAMP)
endif

# Specify debug options for build (default to Release build)
ifeq ($(DEBUG),1)
    BUILD_TYPE := Debug
else
    BUILD_TYPE := Release
endif

# Handle different nvidia-docker version. Do not use nvidia-docker when running with CPUs
ifeq ($(USE_CPU), 1)
    export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):/work/openvino
    DOCKER_RUN_CMD := docker run
else
ifeq ($(USE_INFERENTIA)), 1)
    export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):/work/python_backend
    DOCKER_RUN_CMD := docker run
else
ifneq ($(wildcard /usr/bin/nvidia-docker),)
    DOCKER_RUN_CMD := nvidia-docker run
    # Set Environment variables to fix docker client and server version mismatch
    # Related issue: https://github.com/kubernetes-sigs/kubespray/issues/6160
    export DOCKER_API_VERSION=1.40
else
    DOCKER_RUN_CMD := docker run --gpus=all
endif
endif
endif

# If DOCKER_COMMAND is not pass, launch interactive docker container session.
ifeq ($(DOCKER_COMMAND),)
    DOCKER_INTERACTIVE_FLAGS = -it
else
    DOCKER_INTERACTIVE_FLAGS =
endif

# Default NVIDIA_VISIBLE_DEVICES should use 'all'.
NVIDIA_VISIBLE_DEVICES ?= all

ALLOWED_MIG_CONFS := OFF 1 2 3 ALL
MIG_CONF ?= OFF
ifeq ($(filter $(MIG_CONF),$(ALLOWED_MIG_CONFS)),)
    $(warning MIG_CONF was not set to a valid value. Default to OFF to be safe.)
    MIG_CONF := OFF
endif

# Driver and cuda version check for x86 and aarch64 non-soc system
MIN_DRIVER_VER := 510
ifeq ($(IS_SOC)$(USE_CPU)$(USE_INFERENTIA)$(SKIP_DRIVER_CHECK), 0000)
    DRIVER_VER_MAJOR ?= $(shell nvidia-smi | /bin/grep -Eo 'Driver Version: [+-]?[0-9]+' | awk -F ' ' '{print $$NF}')
    # Check driver version and launch the appropriate container.
    ifeq ($(shell if [ $(DRIVER_VER_MAJOR) -ge $(MIN_DRIVER_VER) ]; then echo true; else echo false; fi), true)
	    CUDA_VER := 11.6
        DRIVER_VER_MAJOR := $(MIN_DRIVER_VER)
    else
        $(error MLPerf Inference v2.0 code requires NVIDIA Driver Version >= $(MIN_DRIVER_VER).xx)
    endif # Driver check
else
    CUDA_VER := 11.6
    DRIVER_VER_MAJOR := $(MIN_DRIVER_VER)
endif

ifneq ($(IS_SOC), 1)
    DOCKER_IMAGE_NAME := base-cuda$(CUDA_VER)-$(ARCH)-ubuntu20.04
    # Check if we are on intranet
    ifeq ($(shell bash $(PROJECT_ROOT)/scripts/check_intranet.sh),0)
        BASE_IMAGE ?= gitlab-master.nvidia.com/compute/mlperf-inference:$(DOCKER_IMAGE_NAME)
    else
        ifeq ($(CUDA_VER), 11.6)
            ifeq ($(TARGET_X86_64), 1)
                BASE_IMAGE ?= nvidia/cuda:11.6.0-devel-ubuntu20.04@sha256:f9a67849a6298e0cbeebfca5c9da8d3891ffa3e8d30bfd667326ffe232d8fc63
            else ifeq ($(TARGET_AARCH64), 1)
                BASE_IMAGE ?= nvidia/cuda:11.6.0-devel-ubuntu20.04@sha256:119c1041e2041f500cc943c1494c5aee587c325bbacede427071387d6aeaf94d
            else
                $(error MLPerf Inference only supports x86 and aarch64 system now.)
            endif
        else
            $(error MLPerf Inference v2.0 code requires cuda version 11.6)
        endif
    endif # check_intranet
endif # xavier check

TEST_FLAGS := --ignore=internal/test_data --ignore=internal/correlation --ignore=internal/__pycache__ -s --durations 0 -vv -rXfw
SM_GENCODE := $(shell ./scripts/get_gencode_$(ARCH) 2> /dev/null)

############################## PREBUILD ##############################
# Build the docker image and launch an interactive container.
# For CPU builds, first build the backend libraries and copy them into the working directory
.PHONY: prebuild
prebuild:
ifeq ($(USE_CPU), 1)
	@$(MAKE) -f $(MAKEFILE_NAME) build_triton_cpu_backends
endif
ifeq ($(USE_INFERENTIA), 1)
	@$(MAKE) -f $(MAKEFILE_NAME) build_triton_inferentia_backends
endif
	@$(MAKE) -f $(MAKEFILE_NAME) build_docker NO_BUILD?=1
ifneq ($(strip ${DOCKER_DETACH}), 1)
	@$(MAKE) -f $(MAKEFILE_NAME) configure_mig MIG_CONF=$(MIG_CONF)
	@$(MAKE) -f $(MAKEFILE_NAME) attach_docker || true
	@$(MAKE) -f $(MAKEFILE_NAME) teardown_mig MIG_CONF=$(MIG_CONF)
endif

# Configure MIG
.PHONY: configure_mig
configure_mig:
	if [ $(MIG_CONF) != "OFF" ]; then MIG_CONF=$(MIG_CONF) ./scripts/mig_configure.sh; fi

# Tear down MIG
.PHONY: teardown_mig
teardown_mig:
	if [ $(MIG_CONF) != "OFF" ]; then ./scripts/mig_teardown.sh; fi

# Clone Triton.
.PHONY: clone_triton
clone_triton:
	@if [ ! -d $(TRITON_DIR) ]; then \
		echo "Cloning Triton Inference Server" \
			&& git clone $(TRITON_URL) $(TRITON_DIR); \
	fi
	@$(eval COMMIT_DISTANCE := $(shell cd $(TRITON_DIR) && git fetch && git rev-list --count origin/master...$(TRITON_HASH)))
	@if [ $(CHECK_TRITON_VERSION) == 1 ]; then \
		if [ $(COMMIT_DISTANCE) -ge 25 ] ; then \
			echo "Error: Triton hash is more than 25 commits behind main. Please update triton" && exit 1; \
		fi \
	fi
	@if [ $(CHECK_TRITON_VERSION) == 1 ]; then \
		if [ $(COMMIT_DISTANCE) -ge 15 ] ; then \
			if [ $(BYPASS_TRITON_WARNING) -lt 1 ] ; then \
				echo "Error: Triton hash is more than 15 commits behind main. Consider updating triton or run with BYPASS_TRITON_WARNING=1" && exit 1; \
			fi \
		fi \
	fi
	@cd $(TRITON_DIR) && git fetch && git checkout $(TRITON_HASH)

# Add symbolic links to scratch path if it exists.
.PHONY: link_dirs
link_dirs:
	@mkdir -p build
ifeq ($(USE_CPU), 1)
	@rm -f openvino
	@ln -sfn openvino_21_04 openvino
ifneq (,$(findstring bert,$(RUN_ARGS)))
ifneq (,$(findstring server,$(RUN_ARGS)))
	@rm -f openvino
	@ln -sfn openvino_21_02 openvino
endif
endif
endif
	@mkdir -p $(DATA_DIR)
	@mkdir -p $(PREPROCESSED_DATA_DIR)
	@mkdir -p $(MODEL_DIR)
	@ln -sfn $(DATA_DIR) $(DATA_DIR_LINK)
	@ln -sfn $(PREPROCESSED_DATA_DIR) $(PREPROCESSED_DATA_DIR_LINK)
	@ln -sfn $(MODEL_DIR) $(MODEL_DIR_LINK)

# Small helper to check if nvidia-docker is installed correctly.
.PHONY: docker_sanity
docker_sanity:
	docker pull nvcr.io/nvidia/cuda:11.0.3-runtime-ubuntu18.04
	$(DOCKER_RUN_CMD) --rm \
		-e NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
		nvcr.io/nvidia/cuda:11.0.3-runtime-ubuntu18.04 nvidia-smi
	@echo "Nvidia-docker is installed correctly!"

# Build the docker image for x86 and aarch64 non-xavier systems.
.PHONY: build_docker
build_docker:
ifeq ($(SUPPORT_DOCKER), 1)
	@echo "Building Docker image"
ifeq ($(NO_DOCKER_PULL), 0)
ifneq ($(USE_CPU), 1)
	docker pull $(BASE_IMAGE)
endif
endif
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build -t mlperf-inference:$(DOCKER_TAG)-latest \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg CUDA_VER=$(CUDA_VER) \
		--build-arg DRIVER_VER_MAJOR=$(DRIVER_VER_MAJOR) \
		--build-arg SM_GENCODE=$(SM_GENCODE) \
		--build-arg USE_CPU=$(USE_CPU) \
		--build-arg USE_NIGHTLY=$(USE_NIGHTLY) \
		--network host \
		-f docker/Dockerfile.$(ARCH) docker
ifeq ($(NO_BUILD), 0)
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build -t mlperf-inference:$(DOCKER_TAG)-latest --no-cache --network host \
		--build-arg BASE_IMAGE=mlperf-inference:$(DOCKER_TAG)-latest \
		-f docker/Dockerfile.build .
endif # NO_BUILD
endif # OS/xavier check

# Add current user into docker image.
.PHONY: docker_add_user
docker_add_user:
ifeq ($(SUPPORT_DOCKER), 1)
	@echo "Adding user account into image"
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build -t mlperf-inference:$(DOCKER_TAG) --network host \
		--build-arg BASE_IMAGE=mlperf-inference:$(DOCKER_TAG)-latest \
		--build-arg GID=$(GROUPID) --build-arg UID=$(UID) --build-arg GROUP=$(GROUPNAME) --build-arg USER=$(UNAME) \
		- < docker/Dockerfile.user
endif

# Add user and launch an interactive container session.
.PHONY: attach_docker
attach_docker:
	@$(MAKE) -f $(MAKEFILE_NAME) docker_add_user
ifneq ($(USE_INFERENTIA), 1)
	@$(MAKE) -f $(MAKEFILE_NAME) launch_docker
endif
ifeq ($(USE_INFERENTIA), 1)
	@$(MAKE) -f $(MAKEFILE_NAME) launch_inferentia_docker
endif

# Launch a container session for inferentia.
.PHONY: launch_inferentia_docker
launch_inferentia_docker:
	@echo "Launching inferentia docker"
	docker run --rm $(DOCKER_INTERACTIVE_FLAGS) -w /work \
	-v $(HOST_VOL):$(CONTAINER_VOL) -v ${HOME}:/${HOME}:rw \
	$(shell scripts/get_inferentia_device_list.sh) \
	--cap-add SYS_ADMIN --cap-add SYS_TIME \
	-e "AWS_NEURON_VISIBLE_DEVICES=ALL" \
	-e "USE_INFERENTIA=1" \
	--shm-size=32gb \
	-v /lib/udev:/mylib/udev \
	-v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro \
	--security-opt apparmor=unconfined --security-opt seccomp=unconfined \
	--name $(DOCKER_NAME) -h $(shell echo $(DOCKER_NAME) | cut -c -64) --add-host $(DOCKER_NAME):127.0.0.1 \
	--cpuset-cpus $(shell taskset -c -p $$$$ | awk '{print $$NF}') \
	--user $(UID):$(GROUPID) --net host --device /dev/fuse \
	--ulimit memlock=-1 \
	$(DOCKER_ARGS) \
	-e MLPERF_SCRATCH_PATH=/home/ubuntu/mlperf_scratch \
	-e HOST_HOSTNAME=$(HOSTNAME) \
	mlperf-inference:$(DOCKER_TAG) $(DOCKER_COMMAND)

# Launch a container session.
.PHONY: launch_docker
launch_docker:
ifeq ($(SUPPORT_DOCKER), 1)
	$(DOCKER_RUN_CMD) --rm $(DOCKER_INTERACTIVE_FLAGS) -w /work \
		-v $(HOST_VOL):$(CONTAINER_VOL) -v ${HOME}:/mnt/${HOME} \
		--cap-add SYS_ADMIN --cap-add SYS_TIME \
		-e NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
		-e HISTFILE=/mnt/${HOME}/.mlperf_bash_history \
		--shm-size=32gb \
		-v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro \
		--security-opt apparmor=unconfined --security-opt seccomp=unconfined \
		--name $(DOCKER_NAME) -h $(shell echo $(DOCKER_NAME) | cut -c -64) --add-host $(DOCKER_NAME):127.0.0.1 \
		--cpuset-cpus $(shell taskset -c -p $$$$ | awk '{print $$NF}') \
		--user $(UID):$(GROUPID) --net host --device /dev/fuse \
		$(DOCKER_MOUNTS) $(DOCKER_ARGS) \
		-e MLPERF_SCRATCH_PATH=$(MLPERF_SCRATCH_PATH) \
		-e HOST_HOSTNAME=$(HOSTNAME) \
		$(shell if [ $(MIG_CONF) == "ALL" ]; then echo "--gpus all -e NVIDIA_MIG_CONFIG_DEVICES=all"; elif [ $(MIG_CONF) != "OFF" ]; then echo "--gpus '\"device=`bash scripts/mig_get_uuid.sh`\"'"; fi) \
		mlperf-inference:$(DOCKER_TAG) $(DOCKER_COMMAND)
endif

############################## DOWNLOAD_MODEL ##############################

BENCHMARKS ?= resnet50 ssd-resnet34 ssd-mobilenet bert dlrm rnnt 3d-unet

.PHONY: download_model
download_model: link_dirs
	$(foreach _benchmark,$(BENCHMARKS),bash code/$(_benchmark)/tensorrt/download_model.sh &&) \
		echo "Finished downloading all the models!"

############################## DOWNLOAD_DATA ##############################

.PHONY: download_data
download_data: link_dirs
	@$(foreach _benchmark,$(BENCHMARKS),bash code/$(_benchmark)/tensorrt/download_data.sh &&) \
		echo "Finished downloading all the datasets!"

############################## PREPROCESS_DATA ##############################

.PHONY: preprocess_data
preprocess_data: link_dirs
	@$(foreach _benchmark,$(BENCHMARKS),$(PYTHON3_CMD) code/$(_benchmark)/tensorrt/preprocess_data.py --data_dir=$(DATA_DIR) --preprocessed_data_dir=$(PREPROCESSED_DATA_DIR) &&) \
		echo "Finished preprocessing all the datasets!"

############################### BUILD GPU ###############################

# Build all source codes.
.PHONY: build
build: clone_loadgen clone_power_dev clone_triton link_dirs
	@$(MAKE) -f $(MAKEFILE_NAME) build_triton
	@$(MAKE) -f $(MAKEFILE_NAME) build_plugins
	@$(MAKE) -f $(MAKEFILE_NAME) build_loadgen
	@$(MAKE) -f $(MAKEFILE_NAME) build_harness
	@$(MAKE) -f $(MAKEFILE_NAME) build_resnet50_plugin

# Clone LoadGen repo.
.PHONY: clone_loadgen
clone_loadgen:
	@if [ ! -d $(LOADGEN_INCLUDE_DIR) ]; then \
		echo "Cloning Official MLPerf Inference (For Loadgen Files)" \
			&& git clone $(INFERENCE_URL) $(INFERENCE_DIR); \
	fi
	@echo "Updating Loadgen" \
		&& cd $(INFERENCE_DIR) \
		&& git fetch \
		&& git checkout $(INFERENCE_HASH) \
		&& git submodule update --init tools/submission/power-dev \
		&& git submodule update --init third_party/pybind \
		&& git submodule update --init language/bert/DeepLearningExamples \
		&& git submodule update --init vision/medical_imaging/3d-unet-brats19/nnUnet

# Clone power-dev repo.
.PHONY: clone_power_dev
clone_power_dev:
	@if [ ! -d $(POWER_DEV_DIR) ]; then \
		echo "Cloning Official Power-Dev repo" \
			&& git clone $(POWER_DEV_URL) $(POWER_DEV_DIR); \
	fi
	@echo "Updating Power-Dev repo" \
		&& cd $(POWER_DEV_DIR) \
		&& git fetch \
		&& git checkout $(POWER_DEV_HASH)

# Build Triton.
.PHONY: build_triton
build_triton:
	@echo "Building TensorRT Inference Server..."
	@if [ ! -d $(TRITON_DIR) ]; then \
		echo "triton-inference-server does not exist! Please exit the container and run make prebuild again." \
			&& exit 1; \
	fi
	# Required till triton build.py properly supports incremental builds
	@mkdir -p $(TRITON_OUT_DIR) \
		&& cd $(TRITON_DIR) \
		&& ./build.py --cmake-dir=$(TRITON_DIR)/build \
			--build-dir=$(TRITON_OUT_DIR) \
			--build-type=$(BUILD_TYPE) \
			--enable-gpu \
			--repo-tag=common:$(TRITON_COMMON_HASH) \
			--repo-tag=core:$(TRITON_CORE_HASH) \
			--repo-tag=backend:$(TRITON_BACKEND_HASH) \
			--repo-tag=thirdparty:$(TRITON_THIRDPARTY_HASH) \
			--backend=tensorrt:$(TRITON_TENSORRT_HASH) \
			--enable-logging \
			--no-container-build
			#--backend=ensemble  --enable-stats --enable-tracing --enable-metrics --enable-gpu-metrics --enable-nvtx

# Build TensorRT plugins.
.PHONY: build_plugins
build_plugins:
	mkdir -p build/plugins/NMSOptPlugin
	cd build/plugins/NMSOptPlugin \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(PROJECT_ROOT)/code/plugin/NMSOptPlugin \
		&& make -j
	# Soc (Xavier, Orin) doesn't need DLRM interaction plugin.
ifneq ($(IS_SOC), 1)
		mkdir -p build/plugins/DLRMInteractionsPlugin
		cd build/plugins/DLRMInteractionsPlugin \
			&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(PROJECT_ROOT)/code/plugin/DLRMInteractionsPlugin \
			&& make -j
endif
	mkdir -p build/plugins/RNNTOptPlugin
	cd build/plugins/RNNTOptPlugin \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(PROJECT_ROOT)/code/plugin/RNNTOptPlugin \
		&& make -j
	mkdir -p build/plugins/pixelShuffle3DPlugin
	cd build/plugins/pixelShuffle3DPlugin \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(PROJECT_ROOT)/code/plugin/pixelShuffle3DPlugin \
		&& make -j
	mkdir -p build/plugins/conv3D1X1X1K4Plugin
	cd build/plugins/conv3D1X1X1K4Plugin \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(PROJECT_ROOT)/code/plugin/conv3D1X1X1K4Plugin \
		&& make -j

# Build resnet50 plugins.
.PHONY: build_resnet50_plugin
build_resnet50_plugin:
	mkdir -p build/plugins/Serial3Conv2dPlugin
	cd build/plugins/Serial3Conv2dPlugin \
                && cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(PROJECT_ROOT)/code/plugin/Serial3Conv2dPlugin \
                && make -j

# Build LoadGen.
.PHONY: build_loadgen
build_loadgen:
	@echo "Building loadgen..."
	@if [ ! -e $(LOADGEN_LIB_DIR) ]; then \
		mkdir $(LOADGEN_LIB_DIR); \
	fi
	@cd $(LOADGEN_LIB_DIR) \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) .. \
		&& make -j

# Build harness source codes.
.PHONY: build_harness
build_harness:
	@echo "Building harness..."
	@mkdir -p build/harness \
		&& cd build/harness \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DIS_SOC=$(IS_SOC) -DSOC_SM=$(SOC_SM) -DLOADGEN_INCLUDE_DIR=$(LOADGEN_INCLUDE_DIR) -DLOADGEN_LIB_DIR=$(LOADGEN_LIB_DIR) $(PROJECT_ROOT)/code/harness \
		&& make -j
	@echo "Finished building harness."

# Compile scripts
build_scripts:
	@nvcc scripts/get_gencode.cu -o scripts/get_gencode_$(ARCH)

#######################  BUILD - TRITON CPU  #######################
# Build required TF and OV backends
.PHONY: build_triton_cpu_backends
build_triton_cpu_backends:
	echo "Building OV Triton backend libraries..."
	./scripts/build_triton_cpu_libs.sh

.PHONY: build_triton_inferentia_backends
build_triton_inferentia_backends:
	echo "Building inferentia Triton backend libraries..."
	./scripts/build_triton_inferentia_libs.sh

# Build all source codes for Triton-CPU.
.PHONY: build_cpu
build_cpu: clone_loadgen clone_power_dev clone_triton link_dirs
	@$(MAKE) -f $(MAKEFILE_NAME) build_triton_cpu
	@$(MAKE) -f $(MAKEFILE_NAME) build_loadgen
	@$(MAKE) -f $(MAKEFILE_NAME) build_harness_cpu

.PHONY: build_inferentia
build_inferentia: clone_loadgen clone_power_dev clone_triton link_dirs
	@$(MAKE) -f $(MAKEFILE_NAME) build_triton_inferentia
	@$(MAKE) -f $(MAKEFILE_NAME) build_loadgen
	@$(MAKE) -f $(MAKEFILE_NAME) build_harness_inferentia

# Build Triton-CPU config
# Note: Triton-CPU only supports x86 now.
.PHONY: build_triton_cpu
build_triton_cpu:
ifeq ($(ARCH), x86_64)
	@echo "Building TensorRT Inference Server..."
	@if [ ! -d $(TRITON_DIR) ]; then \
		echo "triton-inference-server does not exist! Please exit the container and run make prebuild again." \
			&& exit 1; \
	fi
	@mkdir -p $(TRITON_OUT_DIR) \
		&& cd $(TRITON_DIR) \
		&& python3 build.py --cmake-dir=$(TRITON_DIR)/build \
			--build-dir=$(TRITON_OUT_DIR) \
			--enable-gpu \
			--repo-tag=common:$(TRITON_COMMON_HASH) \
			--repo-tag=core:$(TRITON_CORE_HASH) \
			--repo-tag=backend:$(TRITON_BACKEND_HASH) \
			--repo-tag=thirdparty:$(TRITON_THIRDPARTY_HASH) \
			--backend=tensorrt:$(TRITON_TENSORRT_HASH) \
			--enable-logging \
			--no-force-clone \
			--no-container-build
			#--backend=ensemble --enable-stats --enable-tracing --enable-metrics --enable-gpu-metrics --enable-nvtx
endif

.PHONY: build_triton_inferentia
build_triton_inferentia:
ifeq ($(ARCH), x86_64)
	@echo "Building Triton Inference Server..."
	@if [ ! -d $(TRITON_DIR) ]; then \
		echo "triton-inference-server does not exist! Please exit the container and run make prebuild again." \
			&& exit 1; \
	fi
	@mkdir -p $(TRITON_OUT_DIR) \
		&& cd $(TRITON_DIR) \
		&& ./build.py --cmake-dir=$(TRITON_DIR)/build \
			--build-dir=$(TRITON_OUT_DIR) \
			--repo-tag=common:$(TRITON_COMMON_HASH)  \
			--repo-tag=core:$(TRITON_CORE_HASH) \
			--repo-tag=backend:$(TRITON_BACKEND_HASH) \
			--repo-tag=thirdparty:$(TRITON_THIRDPARTY_HASH) \
			--backend=python:$(TRITON_PYTHON_BE_HASH) \
			--enable-logging \
			--no-force-clone \
			--no-container-build
			#--backend=ensemble --enable-stats --enable-tracing --enable-metrics --enable-gpu-metrics --enable-nvtx
	@chmod +x python_backend/inferentia/scripts/setup.sh
	@cd python_backend && \
		./inferentia/scripts/setup.sh -v 3.7 -p -b /work/python_backend -i /work && cd -
	@sudo cp ./python_backend/build/triton_python_backend_stub ./build/triton-inference-server/out/python/build/triton_python_backend_stub
	@sudo cp ./python_backend/build/triton_python_backend_stub ./build/triton-inference-server/out/python/install/backends/python/triton_python_backend_stub
	@sudo cp ./python_backend/build/triton_python_backend_stub ./build/triton-inference-server/out/opt/tritonserver/backends/python/triton_python_backend_stub
endif

# Build Triton-CPU harness source codes.
.PHONY: build_harness_cpu
build_harness_cpu:
	@echo "Building Triton-CPU harness..."
	@mkdir -p build/harness \
		&& cd build/harness \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DLOADGEN_INCLUDE_DIR=$(LOADGEN_INCLUDE_DIR) -DLOADGEN_LIB_DIR=$(LOADGEN_LIB_DIR) -DUSE_CPU=$(USE_CPU) $(PROJECT_ROOT)/code/harness \
		&& make -j
	@echo "Finished building Triton-CPU harness."


# Build Triton-Inferentia harness source codes.
.PHONY: build_harness_inferentia
build_harness_inferentia:
	@echo "Building Triton-Inferentia harness..."
	@mkdir -p build/harness \
		&& cd build/harness \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DLOADGEN_INCLUDE_DIR=$(LOADGEN_INCLUDE_DIR) -DLOADGEN_LIB_DIR=$(LOADGEN_LIB_DIR) -DUSE_INFERENTIA=$(USE_INFERENTIA) $(PROJECT_ROOT)/code/harness \
		&& make -j
	@echo "Finished building Triton-Inferentia harness."

###############################  RUN  ###############################

# Generate TensorRT engines (plan files) and run the harness.
.PHONY: run
run:
	@$(MAKE) -f $(MAKEFILE_NAME) generate_engines
	@$(MAKE) -f $(MAKEFILE_NAME) run_harness

# Generate TensorRT engines (plan files).
.PHONY: generate_engines
generate_engines: link_dirs
	@$(PYTHON3_CMD) code/main.py $(RUN_ARGS) --action="generate_engines"

# Run the harness and check accuracy if in AccuracyOnly mode.
# Add "set -o pipefail" so that "<command> | tee output.txt" will fail when "<command>" fails because otherwise tee
# would return status 0 and clear the failure status.
.PHONY: run_harness
run_harness: link_dirs
	@mkdir -p $(LOG_DIR)
	@set -o pipefail && $(PYTHON3_CMD) code/main.py $(RUN_ARGS) --action="run_harness" 2>&1 | tee $(LOG_DIR)/stdout.txt
	@set -o pipefail && $(PYTHON3_CMD) scripts/print_harness_result.py $(RUN_ARGS) 2>&1 | tee -a $(LOG_DIR)/stdout.txt

# Run CPU-Triton harness. Update LD_LIBRARY_PATH with prebuild backend libraries
.PHONY: run_cpu_harness
run_cpu_harness: link_dirs
	LD_LIBRARY_PATH=/work/prebuilt_triton_libs:$(LD_LIBRARY_PATH) $(PYTHON3_CMD) code/main.py $(RUN_ARGS) --action="run_cpu_harness"
	LD_LIBRARY_PATH=/work/prebuilt_triton_libs:$(LD_LIBRARY_PATH) $(PYTHON3_CMD) scripts/print_harness_result.py $(RUN_ARGS)

.PHONY: run_inferentia_harness
run_inferentia_harness: link_dirs
	LD_LIBRARY_PATH=/work/prebuilt_triton_libs:$(LD_LIBRARY_PATH) $(PYTHON3_CMD) code/main.py $(RUN_ARGS) --action="run_inferentia_harness"
	LD_LIBRARY_PATH=/work/prebuilt_triton_libs:$(LD_LIBRARY_PATH) $(PYTHON3_CMD) scripts/print_harness_result.py $(RUN_ARGS)

# Run the harness and measure power consumption using official MLPerf-Inference power workflow.
.PHONY: run_harness_power
run_harness_power: link_dirs
	@mkdir -p $(POWER_LOGS_DIR)
	@$(MAKE) -f $(MAKEFILE_NAME) power_prologue
	set -o pipefail && $(PYTHON3_CMD) $(POWER_CLIENT_SCRIPT) -a $(POWER_SERVER_IP) -n "$(POWER_NTP_URL)" -S \
		-L $(POWER_LOGS_TEMP_DIR) -o $(POWER_LOGS_DIR) -f \
		-w 'LOG_DIR=$(POWER_LOGS_TEMP_DIR) $(PYTHON3_CMD) code/main.py $(RUN_ARGS) --action="run_harness" \
			2>&1 | tee -a $(POWER_LOGS_DIR)/stdout.txt \
			&& if [ ! -d $(POWER_LOGS_DIR)/ranging_tmp ]; \
				then mkdir $(POWER_LOGS_DIR)/ranging_tmp \
					&& mv $(POWER_LOGS_TEMP_DIR)/* $(POWER_LOGS_DIR)/ranging_tmp/ \
					&& cp -v $(POWER_LOGS_DIR)/ranging_tmp/*/*/*/mlperf_log_detail.txt $(POWER_LOGS_TEMP_DIR)/ \
					&& cp -v $(POWER_LOGS_DIR)/ranging_tmp/*/*/*/mlperf_log_summary.txt $(POWER_LOGS_TEMP_DIR)/; \
				else mkdir $(POWER_LOGS_DIR)/testing_tmp \
					&& mv $(POWER_LOGS_TEMP_DIR)/* $(POWER_LOGS_DIR)/testing_tmp/ \
					&& cp -v $(POWER_LOGS_DIR)/testing_tmp/*/*/*/mlperf_log_detail.txt $(POWER_LOGS_TEMP_DIR)/ \
					&& cp -v $(POWER_LOGS_DIR)/testing_tmp/*/*/*/mlperf_log_summary.txt $(POWER_LOGS_TEMP_DIR)/; fi'
	@$(MAKE) -f $(MAKEFILE_NAME) power_epilogue

.PHONY: run_audit_harness
run_audit_harness: link_dirs
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test01
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test04
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test05

AUDIT_HARNESS := run_audit_harness
ifeq ($(USE_CPU), 1)
	AUDIT_HARNESS := run_cpu_audit_harness
endif

AUDIT_VERIFICATION := run_audit_verification
ifeq ($(USE_CPU), 1)
	AUDIT_VERIFICATION := run_cpu_audit_verification
endif

.PHONY: run_audit_test01_once
run_audit_test01_once: link_dirs
	@$(PYTHON3_CMD) code/main.py $(RUN_ARGS) --audit_test=TEST01 --action="$(AUDIT_HARNESS)"
	@$(PYTHON3_CMD) code/main.py $(RUN_ARGS) --audit_test=TEST01 --action="$(AUDIT_VERIFICATION)"

# TEST01 is sometimes unstable. Try up to two times before failing.
.PHONY: run_audit_test01
run_audit_test01:
	@for i in 1 2; do echo "TEST01 trial $$i" && $(MAKE) -f $(MAKEFILE_NAME) run_audit_test01_once && break; done

.PHONY: run_audit_test04_once
run_audit_test04_once: link_dirs
	@echo "Sleep to reset thermal state before TEST04-A..." && sleep 20
	@$(PYTHON3_CMD) code/main.py $(RUN_ARGS) --audit_test=TEST04-A --action="$(AUDIT_HARNESS)"
	@echo "Sleep to reset thermal state before TEST04-B..." && sleep 20
	@$(PYTHON3_CMD) code/main.py $(RUN_ARGS) --audit_test=TEST04-B --action="$(AUDIT_HARNESS)"
	@$(PYTHON3_CMD) code/main.py $(RUN_ARGS) --audit_test=TEST04-A --action="$(AUDIT_VERIFICATION)"

# TEST04 is so short that it is sometimes unstable. Try up to three times before failing.
.PHONY: run_audit_test04
run_audit_test04:
	@for i in 1 2 3; do echo "TEST04 trial $$i" && $(MAKE) -f $(MAKEFILE_NAME) run_audit_test04_once && break; done

.PHONY: run_audit_test05_once
run_audit_test05_once: link_dirs
	@echo "Sleep to reset thermal state before TEST05..." && sleep 20
	@$(PYTHON3_CMD) code/main.py $(RUN_ARGS) --audit_test=TEST05 --action="$(AUDIT_HARNESS)"
	@$(PYTHON3_CMD) code/main.py $(RUN_ARGS) --audit_test=TEST05 --action="$(AUDIT_VERIFICATION)"

# TEST05 is sometimes unstable. Try up to two times before failing.
.PHONY: run_audit_test05
run_audit_test05:
	@for i in 1 2; do echo "TEST05 trial $$i" && $(MAKE) -f $(MAKEFILE_NAME) run_audit_test05_once && break; done

.PHONY: lock_clocks
lock_clocks:
ifneq ($(IS_SOC), 1)
	@echo "Locking clocks to $(GPUCLK)"
	@sudo nvidia-smi -lgc $(GPUCLK),$(GPUCLK)
else
ifeq ($(SOC_SM), 72)
	@echo "Locking Xavier clocks to max freq (on cl-63 it is 1.377 GHz). "
	@sudo jetson_clocks
endif
endif

.PHONY: free_clocks
free_clocks:
ifneq ($(IS_SOC), 1)
	@echo "Freeing clocks"
	@sudo nvidia-smi -rgc
else
ifeq ($(SOC_SM), 72)
	@echo "Freeing clocks on Xavier is not implemented yet. "
endif
endif

.PHONY: copy_profiles
copy_profiles:
	@echo "Copying yml and sqlite files"
	@cp *.yml /home/scratch.dlsim/data/mlperf-inference/
	@cp *.sqlite /home/scratch.dlsim/data/mlperf-inference/

.PHONY: run_harness_correlation
run_harness_correlation: link_dirs
	@$(MAKE) -f $(MAKEFILE_NAME) lock_clocks
	@$(MAKE) -f $(MAKEFILE_NAME) run_harness || ($(MAKE) -f $(MAKEFILE_NAME) free_clocks ; exit 1)
	@$(MAKE) -f $(MAKEFILE_NAME) free_clocks

# Re-generate TensorRT calibration cache.
.PHONY: calibrate
calibrate: link_dirs
	@$(PYTHON3_CMD) code/main.py $(RUN_ARGS) --action="calibrate"

# When running DLRM benchmark on multi-GPU systems, sometimes it is required to set this to reduce the likelihood of
# out-of-memory issues.
.PHONY: set_virtual_memory_overcommit
set_virtual_memory_overcommit:
	@sudo sysctl -w vm.max_map_count=16777216
	@sudo sysctl -w vm.overcommit_memory=2
	@sudo sysctl -w vm.overcommit_ratio=70

############################## POWER SUBMISSION ##############################
.PHONY: power_prologue
power_prologue:
	@mkdir -p $(POWER_LOGS_TEMP_DIR)
	@mkdir -p $(POWER_LOGS_DIR)
	@rm -rf $(POWER_LOGS_TEMP_DIR)
	@echo "Checking if someone else is using the Power Server machine..."
	@$(MAKE) -f $(MAKEFILE_NAME) power_check_server_not_running
	@echo "Copying Server config to the Power Server..."
	$(POWER_SCP) $(POWER_SERVER_CONFIG) $(POWER_SERVER_USER_IP):$(POWER_SERVER_SCRIPT_DIR)/server.cfg
	@echo "Running Server script on the Power Server..."
	$(POWER_SSH) $(POWER_SERVER_USER_IP) \
		powershell -Command "Start-Process D:\python37\python -PassThru -Wait \
		-ArgumentList '$(POWER_SERVER_SCRIPT_DIR)/server.py -c $(POWER_SERVER_SCRIPT_DIR)/server.cfg' \
		-WorkingDirectory $(POWER_SERVER_SCRIPT_DIR) -WindowStyle Hidden \
		-RedirectStandardOutput $(POWER_SERVER_SCRIPT_DIR)/server_stdout.log \
		-RedirectStandardError $(POWER_SERVER_SCRIPT_DIR)/server.log" &
	@echo "Checking if Power Server script was successfully started..."
	@for i in {1..3}; do $(MAKE) -f $(MAKEFILE_NAME) power_check_server_running && break; done
	@echo "Power Server is running and listening to client now."

.PHONY: power_check_server_running
power_check_server_running:
	@echo "Checking if Power Server is currently running..."
	@$(POWER_SSH) $(POWER_SERVER_USER_IP) \
		powershell -Command "Get-Process python -ErrorAction SilentlyContinue"

.PHONY: power_check_server_not_running
power_check_server_not_running:
	@echo "Checking if Power Server is NOT currently running..."
	@echo "You can run \"POWER_SERVER_IP=$(POWER_SERVER_IP) make power_kill_server\" to kill the job running on the Power Server"
	@! $(POWER_SSH) $(POWER_SERVER_USER_IP) \
		powershell -Command "Get-Process python -ErrorAction SilentlyContinue"

.PHONY: power_kill_server
power_kill_server:
	@echo "Killing server.py on Power Server..."
	@$(POWER_SSH) $(POWER_SERVER_USER_IP) \
		powershell -Command "taskkill /IM python.exe /F"

.PHONY: power_epilogue
power_epilogue:
	@echo "Clean up unnecessary files..."
	@mv $(POWER_LOGS_DIR)/20*/* $(POWER_LOGS_DIR)/
	@rm -rf $(POWER_LOGS_DIR)/20* $(POWER_LOGS_DIR)/*/mlperf_log*
	@mv $(POWER_LOGS_DIR)/ranging_tmp/* $(POWER_LOGS_DIR)/ranging/
	@mv $(POWER_LOGS_DIR)/testing_tmp/* $(POWER_LOGS_DIR)/run_1/
	@rm -rf $(POWER_LOGS_DIR)/ranging_tmp $(POWER_LOGS_DIR)/testing_tmp $(POWER_LOGS_TEMP_DIR)
	@set -o pipefail && LOG_DIR=$(POWER_LOGS_DIR)/run_1 $(PYTHON3_CMD) scripts/print_harness_result.py $(RUN_ARGS) 2>&1 | tee -a $(POWER_LOGS_DIR)/stdout.txt
	@echo "Power logs are located in $(POWER_LOGS_DIR)."

.PHONY: power_warn_exclusive_access
power_warn_exclusive_access:
	@echo "Changing MaxQ/MaxP state on a multi-GPU node will FAIL unless you have full access to the system. That is, you need to have grabbed all the GPUs on this node."

.PHONY: power_warn_reboot_machine
power_warn_reboot_machine:
	@echo "It is recommended to reboot the machine after resetting it from MaxQ state to MaxP state!"

.PHONY: bios_warn_reboot_machine
bios_warn_reboot_machine:
	@echo "Please reboot the machine for BIOS changes to take place."

.PHONY: power_set_maxq_state
power_set_maxq_state:
ifeq ($(HOSTNAME_SHORT), computelab-ro-prod-01)
	@$(MAKE) -f $(MAKEFILE_NAME) power_warn_exclusive_access
	@echo "Setting MaxQ power state on NVIDIA DGX Station A100 machine"
	$(SHELL) power/set_nvlink.sh 0
	sudo nvidia-smi -pl 250
else ifeq ($(HOSTNAME_SHORT), altra-g242-p31-01)
	@$(MAKE) -f $(MAKEFILE_NAME) power_warn_exclusive_access
	@echo "Setting MaxQ power state on A100-PCIe_aarch64x4 machine"
	source power/set_fan_speed.sh && MACHINE_TYPE=GIGABYTE set_fan_auto
	$(SHELL) power/set_nvlink.sh 0
	sudo nvidia-smi -pl 200
else ifeq ($(HOSTNAME_SHORT), ipp1-1469)
	@$(MAKE) -f $(MAKEFILE_NAME) power_warn_exclusive_access
	@echo "Setting MaxQ power state on A100-PCIex8 machine"
	source power/set_fan_speed.sh && MACHINE_TYPE=GIGABYTE set_fan_auto
	$(SHELL) power/set_nvlink.sh 0
	sudo nvidia-smi -pl 200
else ifeq ($(HOSTNAME_SHORT), sjc1-luna-02)
	@$(MAKE) -f $(MAKEFILE_NAME) power_warn_exclusive_access
	@echo "Setting MaxQ power state on NVIDIA DGX A100 machine"
	source power/set_fan_speed.sh && MACHINE_TYPE=LUNA set_fan_manual_speed 30 && sleep 10
	$(SHELL) power/check_switch.sh && sleep 10
	$(SHELL) power/set_nvswitch.sh 0 && sleep 10
	$(SHELL) power/set_nvlink.sh 0 && sleep 10
	sudo nvidia-smi -pm 1
	sudo nvidia-smi -pl 275
else ifeq ($(HOSTNAME_SHORT), computelab-310)
	@echo "Setting MaxQ power state on NVIDIA AGX Xavier"
	sudo /usr/sbin/nvpmodel -f power/nvpmodel-computelab-310.conf -m 8
	sudo /usr/sbin/nvpmodel -d cool
	#These commands require password. Please manually run them.
	#sudo bash -c 'echo "14100000.pcie" > /sys/bus/platform/drivers/tegra-pcie-dw/unbind'
else ifeq ($(HOSTNAME_SHORT), computelab-501)
	@echo "Setting MaxQ power state on NVIDIA Xavier NX"
	sudo /usr/sbin/nvpmodel -f power/nvpmodel-computelab-501.conf -m 9
	sudo /usr/sbin/nvpmodel -d cool
	#These commands require password. Please manually run them.
	#sudo bash -c "echo 5000 > /sys/devices/c250000.i2c/i2c-7/7-0040/iio:device0/crit_current_limit_0"
	#sudo bash -c "echo 5000 > /sys/devices/c250000.i2c/i2c-7/7-0040/iio:device0/warn_current_limit_0"
else ifeq ($(HOSTNAME_SHORT), computelab-502)
	@echo "Setting MaxQ power state on NVIDIA Xavier NX"
	sudo /usr/sbin/nvpmodel -f power/nvpmodel-computelab-502.conf -m 9
	sudo /usr/sbin/nvpmodel -d cool
	#These commands require password. Please manually run them.
	#sudo bash -c "echo 5000 > /sys/devices/c250000.i2c/i2c-7/7-0040/iio:device0/crit_current_limit_0"
	#sudo bash -c "echo 5000 > /sys/devices/c250000.i2c/i2c-7/7-0040/iio:device0/warn_current_limit_0"
else
	@echo "MaxQ state is not supported on this machine!"
	@exit 1
endif

.PHONY: power_set_max_fan_speed_luna
power_set_max_fan_speed_luna:
	@echo "Setting fan speed to MAX on NVIDIA DGX A100 machine"
	source power/set_fan_speed.sh && MACHINE_TYPE=LUNA set_fan_manual_speed 100 && sleep 10

.PHONY: power_set_max_fan_speed_gigabyte
power_set_max_fan_speed_gigabyte:
	@echo "Setting fan speed to MAX on Gigabyte Z54 machine"
	source power/set_fan_speed.sh && MACHINE_TYPE=GIGABYTE set_fan_manual_speed 0xFF

.PHONY: power_set_max_fan_speed
power_set_max_fan_speed:
	@$(MAKE) -f $(MAKEFILE_NAME) power_warn_exclusive_access
ifeq ($(HOSTNAME_SHORT), ipp1-1468)
	@$(MAKE) -f $(MAKEFILE_NAME) power_set_max_fan_speed_gigabyte
else ifeq ($(HOSTNAME_SHORT), ipp1-1469)
	@$(MAKE) -f $(MAKEFILE_NAME) power_set_max_fan_speed_gigabyte
else ifeq ($(HOSTNAME_SHORT), ipp1-1470)
	@$(MAKE) -f $(MAKEFILE_NAME) power_set_max_fan_speed_gigabyte
else ifeq ($(HOSTNAME_SHORT), altra-g242-p31-01)
	@$(MAKE) -f $(MAKEFILE_NAME) power_set_max_fan_speed_gigabyte
else ifeq ($(HOSTNAME_SHORT), sjc1-luna-01)
	@$(MAKE) -f $(MAKEFILE_NAME) power_set_max_fan_speed_luna
else ifeq ($(HOSTNAME_SHORT), sjc1-luna-02)
	@$(MAKE) -f $(MAKEFILE_NAME) power_set_max_fan_speed_luna
else ifeq ($(HOSTNAME_SHORT), luna-prod-48)
	@$(MAKE) -f $(MAKEFILE_NAME) power_set_max_fan_speed_luna
else
	@echo "Setting fan speed is not supported on this machine"
endif

.PHONY: power_reset_maxq_state
power_reset_maxq_state:
ifeq ($(HOSTNAME_SHORT), computelab-ro-prod-01)
	@$(MAKE) -f $(MAKEFILE_NAME) power_warn_exclusive_access
	@echo "Resetting power state on NVIDIA DGX Station A100 machine"
	$(SHELL) power/set_nvlink.sh 1
	sudo nvidia-smi -pl 275
	@$(MAKE) -f $(MAKEFILE_NAME) power_warn_reboot_machine
else ifeq ($(HOSTNAME_SHORT), altra-g242-p31-01)
	@$(MAKE) -f $(MAKEFILE_NAME) power_warn_exclusive_access
	@echo "Resetting power state on A100-PCIe_aarch64x4 machine"
	$(SHELL) power/set_nvlink.sh 1
	@$(MAKE) -f $(MAKEFILE_NAME) power_set_max_fan_speed
	sudo nvidia-smi -pl 250
	@$(MAKE) -f $(MAKEFILE_NAME) power_warn_reboot_machine
else ifeq ($(HOSTNAME_SHORT), ipp1-1469)
	@$(MAKE) -f $(MAKEFILE_NAME) power_warn_exclusive_access
	@echo "Resetting power state on A100-PCIex8 machine"
	$(SHELL) power/set_nvlink.sh 1
	@$(MAKE) -f $(MAKEFILE_NAME) power_set_max_fan_speed
	sudo nvidia-smi -pl 250
	@$(MAKE) -f $(MAKEFILE_NAME) power_warn_reboot_machine
else ifeq ($(HOSTNAME_SHORT), sjc1-luna-02)
	@$(MAKE) -f $(MAKEFILE_NAME) power_warn_exclusive_access
	@echo "Resetting power state on NVIDIA DGX A100 machine"
	@$(MAKE) -f $(MAKEFILE_NAME) power_set_max_fan_speed
	$(SHELL) power/set_nvlink.sh 1 && sleep 10
	$(SHELL) power/set_nvswitch.sh 1 && sleep 10
	sudo nvidia-smi -pm 1
	sudo nvidia-smi -pl 400
	sudo systemctl start nvidia-fabricmanager
	sudo systemctl start dcgm
	sudo systemctl start nvsm
	sudo systemctl start nvidia-persistenced.service
	@$(MAKE) -f $(MAKEFILE_NAME) power_warn_reboot_machine
else ifeq ($(HOSTNAME_SHORT), computelab-310)
	@echo "Resetting power state on NVIDIA AGX Xavier"
	@sudo /usr/sbin/nvpmodel -f power/nvpmodel-computelab-310.conf -m 0
	@sudo /usr/sbin/nvpmodel -d cool
else ifeq ($(HOSTNAME_SHORT), computelab-501)
	@echo "Resetting power state on NVIDIA Xavier NX"
	@sudo /usr/sbin/nvpmodel -f power/nvpmodel-computelab-501.conf -m 6
	@sudo /usr/sbin/nvpmodel -d cool
	#These commands require password. Please manually run them.
	#sudo bash -c "echo 5000 > /sys/devices/c250000.i2c/i2c-7/7-0040/iio:device0/crit_current_limit_0"
	#sudo bash -c "echo 5000 > /sys/devices/c250000.i2c/i2c-7/7-0040/iio:device0/warn_current_limit_0"
else ifeq ($(HOSTNAME_SHORT), computelab-502)
	@echo "Resetting power state on NVIDIA Xavier NX"
	@sudo /usr/sbin/nvpmodel -f power/nvpmodel-computelab-502.conf -m 6
	@sudo /usr/sbin/nvpmodel -d cool
	#These commands require password. Please manually run them.
	#sudo bash -c "echo 5000 > /sys/devices/c250000.i2c/i2c-7/7-0040/iio:device0/crit_current_limit_0"
	#sudo bash -c "echo 5000 > /sys/devices/c250000.i2c/i2c-7/7-0040/iio:device0/warn_current_limit_0"
else ifeq ($(HOSTNAME_SHORT), orin-agx-02)
	@echo "Resetting power state on NVIDIA Orin"
	@sudo /usr/sbin/nvpmodel -f power/concord_nvpmodel.conf -m 6
else
	@echo "$(HOSTNAME_SHORT) does not have MaxQ submission."
endif

BIOS_SETTING_HOSTS := computelab-ro-prod-01 ipp1-1469 sjc1-luna-02
.PHONY: power_set_maxq_bios
power_set_maxq_bios: bios_warn_reboot_machine
ifeq ($(HOSTNAME_SHORT), $(filter $(HOSTNAME_SHORT), $(BIOS_SETTING_HOSTS)))
	@echo "Setting BIOS state on $(HOSTNAME_SHORT)"
	source power/bios_setting.sh && set_4_ccd
else
	@echo "Setting MaxQ BIOS state is not supported on this machine."
endif

.PHONY: power_reset_maxq_bios
power_reset_maxq_bios: bios_warn_reboot_machine
ifeq ($(HOSTNAME_SHORT), $(filter $(HOSTNAME_SHORT), $(BIOS_SETTING_HOSTS)))
	@echo "Resetting BIOS state on $(HOSTNAME_SHORT)"
	source power/bios_setting.sh && set_all_ccd
else
	@echo "Resetting MaxQ BIOS state is not supported on this machine."
endif

.PHONY: power_server_setup
power_server_setup:
	scp -o "StrictHostKeyChecking no" power/internal/add_user.ps1 $(UNAME)@$(POWER_SERVER_IP):C:/add_user.ps1
	ssh -o "StrictHostKeyChecking no" $(UNAME)@$(POWER_SERVER_IP) powershell -file C:/add_user.ps1
	$(POWER_SCP) -r $(MLPERF_SCRATCH_PATH)/mlperf_power/PTD $(POWER_SERVER_USER_IP):C:/PTD
	$(POWER_SCP) -r $(MLPERF_SCRATCH_PATH)/mlperf_power/power-dev $(POWER_SERVER_USER_IP):C:/power-dev
	$(POWER_SCP) $(MLPERF_SCRATCH_PATH)/mlperf_power/python-3.7.9-amd64.exe $(POWER_SERVER_USER_IP):C:/python-3.7.9-amd64.exe
	$(POWER_SCP) power/internal/install_power_dependencies.ps1 $(POWER_SERVER_USER_IP):C:/install_power_dependencies.ps1
	$(POWER_SSH) $(POWER_SERVER_USER_IP) powershell -file C:/install_power_dependencies.ps1

.PHONY: power_server_update_power_dev
power_server_update_power_dev:
	$(POWER_SSH) $(POWER_SERVER_USER_IP) 'rd /s /q "C:/power-dev"'
	$(POWER_SCP) -r $(MLPERF_SCRATCH_PATH)/mlperf_power/power-dev $(POWER_SERVER_USER_IP):C:/power-dev

############################## AUTOMATION AND SUBMISSION ##############################
# RUN_ID: ID for the run. For L1, this is the number of the run
# SYSTEM_NAME: Name of the current platform
# ARTIFACT_SRC_PATH: path/to/directory to compress
# ARTIFACT_NAME: name-of-artifact-to-push
# ARTIFACT_DST_PATH: path/in/artifactory/to/prepend
ARTIFACTORY_URL := https://urm.nvidia.com/artifactory/sw-mlpinf-generic

RUN_ID ?= manual-$(TIMESTAMP)
ARTIFACT_NAME ?= $(SYSTEM_NAME)_$(RUN_ID)
ARTIFACT_DST_PATH ?= artifacts

ARTIFACT_URL := $(ARTIFACTORY_URL)/$(ARTIFACT_DST_PATH)/$(ARTIFACT_NAME).gz

# Generate a raw results directory in build/full_results from LoadGen logs in build/logs
.PHONY: update_results
update_results:
	@$(PYTHON3_CMD) scripts/update_results.py --output_dir results --result_id $(ARTIFACT_NAME)
	@printf "If you would like to push results, run:\n\tmake truncate_results\n\tmake push_full_results ARTIFACT_NAME=$(ARTIFACT_NAME)\n"

.PHONY: update_compliance
update_compliance:
	@$(PYTHON3_CMD) scripts/update_results.py --input_dir build/compliance_logs --output_dir compliance --assume_compliance --result_id $(ARTIFACT_NAME)
	@printf "If you would like to push results, run:\n\tmake truncate_results\n\tmake push_full_results ARTIFACT_NAME=$(ARTIFACT_NAME)\n"

.PHONY: truncate_results
truncate_results:
	@echo "WARNING: This script cannot be executed from within the docker container."
	@echo "It must have access to the project root at ../../"
	@rm -rf build/full_results
	@cd ../../ \
		&& $(PYTHON3_CMD) closed/$(SUBMITTER)/build/inference/tools/submission/truncate_accuracy_log.py --input . --backup closed/$(SUBMITTER)/build/full_results --submitter $(SUBMITTER)
	@echo "Full accuracy logs stored in build/full_results/. Truncated results stored in results/."

.PHONY: summarize_results
summarize_results:
	@$(PYTHON3_CMD) scripts/internal/results_analysis/summarize_results.py $(RUN_ARGS)

.PHONY: check_submission
check_submission:
	@echo "WARNING: This script cannot be executed from within the docker container."
	@echo "It must have access to the project root at ../../"
	@cd ../../ \
		&& $(PYTHON3_CMD) closed/$(SUBMITTER)/build/inference/tools/submission/submission-checker.py --input . --submitter $(SUBMITTER) 2>&1 \
		| tee closed/$(SUBMITTER)/results/submission_checker_log.txt

.PHONY: check_submission_power
check_submission_power:
	@cd ../../ \
		&& $(PYTHON3_CMD) closed/$(SUBMITTER)/build/inference/tools/submission/submission-checker.py --more-power-check --input . --submitter $(SUBMITTER) 2>&1 \
		| tee closed/$(SUBMITTER)/results/submission_checker_log.txt

.PHONY: push_artifacts
push_artifacts:
	@mkdir -p build/artifacts
	@tar -cvzf build/artifacts/$(ARTIFACT_NAME).gz $(ARTIFACT_SRC_PATH)
	curl -u$(UNAME):$(ARTIFACTORY_API_KEY) -T build/artifacts/$(ARTIFACT_NAME).gz "$(ARTIFACT_URL)"

.PHONY: push_full_results
push_full_results:
	@$(MAKE) -f $(MAKEFILE_NAME) push_artifacts ARTIFACT_SRC_PATH=build/full_results ARTIFACT_DST_PATH=full_result_logs ARTIFACT_NAME=full-results_$(ARTIFACT_NAME)

############################## UTILITY ##############################

.PHONY: generate_conf_files
generate_conf_files:
	@$(PYTHON3_CMD) scripts/create_config_files.py

.PHONY: autotune
autotune:
	@$(PYTHON3_CMD) scripts/autotune/grid.py $(RUN_ARGS)

# Remove build directory.
.PHONY: clean
clean: clean_shallow
	rm -rf build

# Remove only the files necessary for a clean build.
.PHONY: clean_shallow
clean_shallow:
	rm -rf build/bin
	rm -rf build/harness
	rm -rf build/plugins
	rm -rf $(TRITON_OUT_DIR)
	rm -rf $(LOADGEN_LIB_DIR)
	rm -rf prebuild_triton_libs # Triton CPU libraries

.PHONY: clean_triton
clean_triton:
	rm -rf $(TRITON_DIR)

# Print out useful information.
.PHONY: info
info:
	@echo "RUN_ID=$(RUN_ID)"
	@echo "SYSTEM_NAME=$(SYSTEM_NAME)"
	@echo "Architecture=$(ARCH)"
	@echo "SM_GENCODE=$(SM_GENCODE)"
	@echo "User=$(UNAME)"
	@echo "UID=$(UID)"
	@echo "HOSTNAME=$(HOSTNAME)"
	@echo "Usergroup=$(GROUPNAME)"
	@echo "GroupID=$(GROUPID)"
	@echo "Docker info: {DETACH=$(DOCKER_DETACH), TAG=$(DOCKER_TAG)}"
ifdef DOCKER_IMAGE_NAME
	@echo "Docker image used: $(DOCKER_IMAGE_NAME) -> [$(BASE_IMAGE)]"
endif
	@echo "PYTHON3_CMD=$(PYTHON3_CMD)"
	@echo "PATH=$(PATH)"
	@echo "CPATH=$(CPATH)"
	@echo "CUDA_PATH=$(CUDA_PATH)"
	@echo "LIBRARY_PATH=$(LIBRARY_PATH)"
	@echo "LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)"
	@echo "MIG_CONF=$(MIG_CONF)"

# The shell target will start a shell that inherits all the environment
# variables set by this Makefile for convenience.
.PHONY: shell
shell:
	@$(SHELL)

.PHONY: regression_presubmit
regression_presubmit:
	@$(PYTHON3_CMD) -m pytest $(TEST_FLAGS) --ignore=internal/e2e internal # All unit tests (non-End-to-End)
	@$(PYTHON3_CMD) -m pytest internal/e2e -rXfw -k "regression"
ifeq ($(MINIMAL_REGRESSION_PRESUBMIT), 0)
		@$(PYTHON3_CMD) -m pytest $(TEST_FLAGS) internal/e2e -k "generate_engines"
		@$(PYTHON3_CMD) -m pytest $(TEST_FLAGS) internal/e2e -k "run_harness"
		@$(PYTHON3_CMD) -m pytest $(TEST_FLAGS) internal/e2e -k "accuracy"
endif
	@$(PYTHON3_CMD) -m pytest $(TEST_FLAGS) internal/e2e -k "functionality"
