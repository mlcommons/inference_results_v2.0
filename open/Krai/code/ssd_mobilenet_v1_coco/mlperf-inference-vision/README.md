# MLPerf Inference Vision - extended for Object Detection

This Collective Knowledge workflow is based on the [official MLPerf Inference Vision application](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) extended for diverse Object Detection models, as found e.g. in the [TF1 Object Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) and the [TF2 Object Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).

### Supported Model-Engine-Backend Combinations

The table below shows currently supported models, frameworks ("inference engines") and library/device combinations ("inference engine backends").

| `MODEL_NAME`                                 | `INFERENCE_ENGINE`  | `INFERENCE_ENGINE_BACKEND`                 |
| -------------------------------------------- | ------------------- | ------------------------------------------ |
| `rcnn-nas-lowproposals-coco`                 | `tensorflow`        | `default-cpu`,`default-gpu`,`openvino-cpu` |
| `rcnn-resnet50-lowproposals-coco`            | `tensorflow`        | `default-cpu`,`default-gpu`,`openvino-cpu` |
| `rcnn-resnet101-lowproposals-coco`           | `tensorflow`        | `default-cpu`,`default-gpu`,`openvino-cpu` |
| `rcnn-inception-resnet-v2-lowproposals-coco` | `tensorflow`        | `default-cpu`,`default-gpu`,`openvino-cpu` |
| `rcnn-inception-v2-coco`                     | `tensorflow`        | `default-cpu`,`default-gpu`,`openvino-cpu` |
| `ssd-inception-v2-coco`                      | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssd_mobilenet_v1_coco`                      | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssd_mobilenet_v1_quantized_coco`            | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssd-mobilenet-v1-fpn-sbp-coco`              | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssd-resnet50-v1-fpn-sbp-coco`               | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssdlite-mobilenet-v2-coco`                  | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `yolo-v3-coco`                               | `tensorflow`        | `default-cpu`,`default-gpu`,`openvino-cpu` |
| `ssd_resnet50_v1_fpn_640x640`                | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssd_resnet50_v1_fpn_1024x1024`              | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssd_resnet101_v1_fpn_640x640`               | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssd_resnet101_v1_fpn_1024x1024`             | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssd_resnet152_v1_fpn_640x640`               | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssd_resnet152_v1_fpn_1024x1024`             | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssd_mobilenet_v2_320x320`                   | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssd_mobilenet_v1_fpn_640x640`               | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssd_mobilenet_v2_fpnlite_320x320`           | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssd_mobilenet_v2_fpnlite_640x640`           | `tensorflow`        | `default-cpu`,`default-gpu`                |


### Supported Backend-Scenario-BatchSize Combinations
<details>
<summary>Click to expand</summary>
(to be updated)

|  | tensorflow-v2.7.1-cpu |  |  |  |  |  |  | tensorflow-v2.7.1-gpu|  |  |  |  |  |  | tensorflow-tensorrt-dynamic |  |  |  |  |  |  | tensorflow-openvino-cpu |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  | Single Stream | Single Stream | Offline | Offline | Offline | Offline |  | Single Stream | Single Stream | Offline | Offline | Offline | Offline |  | Single Stream | Single Stream | Offline | Offline | Offline | Offline |  | Single Stream | Single Stream | Offline | Offline | Offline | Offline |
|  | Accuracy | Performance | Accuracy | Performance | Accuracy | Performance |  | Accuracy | Performance | Accuracy | Performance | Accuracy | Performance |  | Accuracy | Performance | Accuracy | Performance | Accuracy | Performance |  | Accuracy | Performance | Accuracy | Performance | Accuracy | Performance |
|  | Batch Size 1 | Batch Size 1 | Batch Size 1 | Batch Size 1 | Batch Size >1 | Batch Size >1 |  | Batch Size 1 | Batch Size 1 | Batch Size 1 | Batch Size 1 | Batch Size >1 | Batch Size >1 |  | Batch Size 1 | Batch Size 1 | Batch Size 1 | Batch Size 1 | Batch Size >1 | Batch Size >1 |  | Batch Size 1 | Batch Size 1 | Batch Size 1 | Batch Size 1 | Batch Size >1 | Batch Size >1 |
| rcnn-nas-lowproposals-coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |
| rcnn-resnet50-lowproposals-coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |
| rcnn-resnet101-lowproposals-coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |
| rcnn-inception-resnet-v2-lowproposals-coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |
| rcnn-inception-v2-coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |
| ssd-inception-v2-coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd_mobilenet_v1_coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd_mobilenet_v1_quantized_coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd-mobilenet-v1-fpn-sbp-coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd-resnet50-v1-fpn-sbp-coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssdlite-mobilenet-v2-coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| yolo-v3-coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |
| ssd_resnet50_v1_fpn_640x640 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd_resnet50_v1_fpn_1024x1024 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd_resnet101_v1_fpn_640x640 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd_resnet101_v1_fpn_1024x1024 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd_resnet152_v1_fpn_640x640 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd_resnet152_v1_fpn_1024x1024 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd_mobilenet_v2_320x320 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd_mobilenet_v1_fpn_640x640 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd_mobilenet_v2_fpnlite_320x320 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd_mobilenet_v2_fpnlite_640x640 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |

🟩 Supported
🟥 Not supported

</details>
<br>

---
<br>

# A) Set Up
## 1. Building the Docker image

In the following examples, TensorFlow 2.7.1 and NVIDIA container image for TensorRT 21.08 were used.
**NB:** The
[TensorRT 21.06](https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel_21-06.html#rel_21-06)
release is the last one to support TensorRT 7.2, needed by TensorFlow 2.7.

```
export CK_IMAGE_NAME=mlperf-inference-vision SDK_VER=21.08-py3 TF_VER=2.7.1
cd $(ck find program:$CK_IMAGE_NAME) && ./build.sh
```

<details>
<summary>Click to expand</summary>

```
Successfully built 9c39ebef9ad2
Successfully tagged krai/mlperf-inference-vision:21.08-py3_tf-2.7.1

real    14m29.990s
user    0m10.826s
sys     0m11.604s

Done.
```
</details>

Set an environment variable for the built image and validate:

```
export CK_IMAGE="krai/${CK_IMAGE_NAME}:${SDK_VER}_tf-${TF_VER}"
docker image ls ${CK_IMAGE}
```

```
REPOSITORY                     TAG                  IMAGE ID       CREATED         SIZE
krai/mlperf-inference-vision   21.08-py3_tf-2.7.1   362d3cd6ddd5   8 minutes ago   16.6GB
```


## 2. Save experimental results into a host directory

The user should belong to the group `krai` on the host machine.
If it does not exist:

```
sudo groupadd krai
sudo usermod -aG krai $USER
```

### Create a new repository

```
export CK_EXPERIMENT_REPO="mlperf_v2.0.object-detection.$(hostname).$(id -un)"
ck add repo:${CK_EXPERIMENT_REPO} --quiet && \
ck add ${CK_EXPERIMENT_REPO}:experiment:dummy --common_func && \
ck rm  ${CK_EXPERIMENT_REPO}:experiment:dummy --force
```

### Make its `experiment` directory writable by group `krai`

```
export CK_EXPERIMENT_DIR="${HOME}/CK/${CK_EXPERIMENT_REPO}/experiment"
chgrp -R krai $CK_EXPERIMENT_DIR && \
chmod -R g+ws $CK_EXPERIMENT_DIR && \
setfacl -R -d -m group:krai:rwx $CK_EXPERIMENT_DIR
```

---
<br>

# B) Run
General Form:
```
ck run cmdgen:benchmark.mlperf-inference-vision --verbose --docker --docker_image=${CK_IMAGE} --experiment_dir=${CK_EXPERIMENT_DIR} --sut=chai 

# Model Specifications
--model=[MODEL_NAME]

# Backend Specifications
--library=[INFERENCE_ENGINE-INFERENCE_ENGINE_BACKEND] --device_ids=[DEVICE]

# Scenario and Mode Specifications
--scenario=[SCENARIO] --mode=[MODE]

# Experiments Specifications
--dataset_size=50 --buffer_size=5 --batch_size=1

# Optional, for SingleStream-Performance only
--target_latency=35

# Optional, for Offline-Performance only
--target_qps=3
```

## 1. Specify the Model
Example of `ssd_resnet50_v1_fpn_640x640` model:
```
ck run cmdgen:benchmark.mlperf-inference-vision --verbose --docker --docker_image=${CK_IMAGE} --experiment_dir=${CK_EXPERIMENT_DIR} --sut=chai \
--model=rcnn-nas-lowproposals-coco \
--library=tensorflow-v2.7.1-gpu --device_ids=0 \
--scenario=singlestream --mode=accuracy \
--dataset_size=50 --buffer_size=5 --batch_size=1
```

## 2. Specify the Scenario and Mode
|Scenario | Mode|
|---|---|
| `SingleStream`, <br> `Offline` | `Accuracy`, <br> `Performance` |

And `batch_size` could be more than 1 for some models in `Offline` Mode.

### Examples
<details>
<summary>Click to expand</summary>

Example of `SingleStream` - `Accuracy` - `Batch Size 1`
```
ck run cmdgen:benchmark.mlperf-inference-vision --verbose --docker --docker_image=${CK_IMAGE} --experiment_dir=${CK_EXPERIMENT_DIR} --sut=chai \
--model=rcnn-nas-lowproposals-coco \
--library=tensorflow-v2.7.1-gpu --device_ids=0 \
--scenario=singlestream --mode=accuracy \
--dataset_size=50 --buffer_size=5 --batch_size=1
```

Example of `SingleStream` - `Performance` - `Batch Size 1`
```
ck run cmdgen:benchmark.mlperf-inference-vision --verbose --docker --docker_image=${CK_IMAGE} --experiment_dir=${CK_EXPERIMENT_DIR} --sut=chai \
--model=rcnn-nas-lowproposals-coco \
--library=tensorflow-v2.7.1-gpu --device_ids=0 \
--scenario=singlestream --mode=performance \
--dataset_size=50 --buffer_size=5 --batch_size=1 --target_latency=35
```

Example of `Offline` - `Accuracy` - `Batch Size 1`
```
ck run cmdgen:benchmark.mlperf-inference-vision --verbose --docker --docker_image=${CK_IMAGE} --experiment_dir=${CK_EXPERIMENT_DIR} --sut=chai \
--model=rcnn-nas-lowproposals-coco \
--library=tensorflow-v2.7.1-gpu --device_ids=0 \
--scenario=offline --mode=accuracy \
--dataset_size=50 --buffer_size=5 --batch_size=1
```

Example of `Offline` - `Performance` - `Batch Size 1`
```
ck run cmdgen:benchmark.mlperf-inference-vision --verbose --docker --docker_image=${CK_IMAGE} --experiment_dir=${CK_EXPERIMENT_DIR} --sut=chai \
--model=rcnn-nas-lowproposals-coco \
--library=tensorflow-v2.7.1-gpu --device_ids=0 \
--scenario=offline --mode=performance \
--dataset_size=50 --buffer_size=5 --batch_size=1 --target_qps=3
```

Example of `Offline` - `Accuracy` - `Batch Size 32`
```
ck run cmdgen:benchmark.mlperf-inference-vision --verbose --docker --docker_image=${CK_IMAGE} --experiment_dir=${CK_EXPERIMENT_DIR} --sut=chai \
--model=rcnn-nas-lowproposals-coco \
--library=tensorflow-v2.7.1-gpu --device_ids=0 \
--scenario=offline --mode=accuracy \
--dataset_size=50 --buffer_size=5 --batch_size=32
```

Example of `Offline` - `Performance` - `Batch Size 32`
```
ck run cmdgen:benchmark.mlperf-inference-vision --verbose --docker --docker_image=${CK_IMAGE} --experiment_dir=${CK_EXPERIMENT_DIR} --sut=chai \
--model=rcnn-nas-lowproposals-coco \
--library=tensorflow-v2.7.1-gpu --device_ids=0 \
--scenario=offline --mode=performance \
--dataset_size=50 --buffer_size=5 --batch_size=32 --target_qps=3
```
</details>
<br>

## 3. Select the Engine/Backend/Device

### Supported `INFERENCE_ENGINE`/`INFERENCE_ENGINE_BACKEND`/`CUDA_VISIBLE_DEVICES` combinations

| `INFERENCE_ENGINE` | `INFERENCE_ENGINE_BACKEND`  | `CUDA_VISIBLE_DEVICES`       |
| ------------------ | --------------------------- | ---------------------------- |
| `tensorflow-v2.7.1`| `cpu`                       | `-1`                         |
| `tensorflow-v2.7.1`| `gpu`                       | `<device_id>`                |
| `tensorflow`       | `tensorrt-dynamic`          | `<device_id>`                |
| `tensorflow`       | `openvino-cpu`              | `-1`                         |

### Examples
<details>
<summary>Click to expand</summary>

Example of `tensorflow-v2.7.1-cpu`
```
ck run cmdgen:benchmark.mlperf-inference-vision --verbose --docker --docker_image=${CK_IMAGE} --experiment_dir=${CK_EXPERIMENT_DIR} --sut=chai \
--model=rcnn-nas-lowproposals-coco \
--library=tensorflow-v2.7.1-cpu --device_ids=-1 \
--scenario=singlestream --mode=accuracy \
--dataset_size=50 --buffer_size=5 --batch_size=1
```

Example of `tensorflow-v2.7.1-gpu`
```
ck run cmdgen:benchmark.mlperf-inference-vision --verbose --docker --docker_image=${CK_IMAGE} --experiment_dir=${CK_EXPERIMENT_DIR} --sut=chai \
--model=rcnn-nas-lowproposals-coco \
--library=tensorflow-v2.7.1-gpu --device_ids=0 \
--scenario=singlestream --mode=accuracy \
--dataset_size=50 --buffer_size=5 --batch_size=1
```

Example of `tensorflow-tensorrt-dynamic`
```
ck run cmdgen:benchmark.mlperf-inference-vision --verbose --docker --docker_image=${CK_IMAGE} --experiment_dir=${CK_EXPERIMENT_DIR} --sut=chai \
--model=rcnn-nas-lowproposals-coco \
--library=tensorflow-tensorrt-dynamic --device_ids=0 \
--scenario=singlestream --mode=accuracy \
--dataset_size=50 --buffer_size=5 --batch_size=1
```

Example of `tensorflow-openvino-cpu`
```
ck run cmdgen:benchmark.mlperf-inference-vision --verbose --docker --docker_image=${CK_IMAGE} --experiment_dir=${CK_EXPERIMENT_DIR} --sut=chai \
--model=rcnn-nas-lowproposals-coco \
--library=tensorflow-openvino-cpu --device_ids=-1 \
--scenario=singlestream --mode=accuracy \
--dataset_size=50 --buffer_size=5 --batch_size=1
```

</details>
<br>
<br>
<br>

More details on how to run the program without `cmdgen` could be found from [README.raw.md](README.raw.md) and more details on the mapping performed by `cmdgen` could be found from [README.cmdgen.md](README.cmdgen.md). 
