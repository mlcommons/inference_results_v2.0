# MLPerf Inference - Image Classification - TFLite

## SingleStream

- Set up [`program:mlperf-inference-vision`](https://github.com/krai/ck-mlperf/blob/master/program/mlperf-inference-vision/README.md) on your SUT.
- Customize the examples below for your SUT. The SUT used below was `chai`.

### Workloads
Support models from TensorFlow Detection Model Zoos: [tf1-zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) , [tf2-zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/)

- rcnn-nas-lowproposals-coco
- rcnn-resnet50-lowproposals-coco
- rcnn-resnet101-lowproposals-coco
- rcnn-inception-resnet-v2-lowproposals-coco
- rcnn-inception-v2-coco
- ssd-inception-v2-coco
- ssd_mobilenet_v1_coco
- ssd_mobilenet_v1_quantized_coco
- ssd-mobilenet-v1-fpn-sbp-coco
- ssd-resnet50-v1-fpn-sbp-coco
- ssdlite-mobilenet-v2-coco
- yolo-v3-coco
- ssd_resnet50_v1_fpn_640x640
- ssd_resnet50_v1_fpn_1024x1024
- ssd_resnet101_v1_fpn_640x640
- ssd_resnet101_v1_fpn_1024x1024
- ssd_resnet152_v1_fpn_640x640
- ssd_resnet152_v1_fpn_1024x1024
- ssd_mobilenet_v2_320x320
- ssd_mobilenet_v1_fpn_640x640
- ssd_mobilenet_v2_fpnlite_320x320
- ssd_mobilenet_v2_fpnlite_640x640

### General

Specifying `--group.closed` runs the benchmark in the following modes required for the Closed division:
- Accuracy with the given `--dataset_size`.
- Performance with the given `--target_latency`.
- Compliance tests (TEST01, TEST04-A/B, TEST05) with the given `--target_latency`.

Specifying `--group.open` runs the benchmark in the following modes required for the Open division:
- Accuracy with the given `--dataset_size`.
- Performance with the given `--target_latency`.

```
python3 -m pip install ck==2.5.8

export CONTAINER_ID=`ck run cmdgen:benchmark.mlperf-inference-vision --docker=container_only --out=none  --library=tensorflow-v2.7.1-gpu --docker_image=${CK_IMAGE} --experiment_dir`
```

### Running All Models 

Running all the models in `tf1-zoo` or `tf2-zoo` for both `accuracy` and `performance` mode. 

### For `tf1-zoo` models
```
time ck run cmdgen:benchmark.mlperf-inference-vision --verbose --sut=chai --model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tf2-zoo --separator=:):$(ck list_variations misc --query_module_uoa=package --tags=model,tf1-zoo --separator=:) --library=tensorflow-v2.7.1-gpu --device_ids=0 --scenario=singlestream --group.open --dataset_size=5000  --batch_size=1 --target_latency_file=/home/krai/CK_REPOS/ck-mlperf/program/mlperf-inference-vision/target_latency.chai.txt --container=$CONTAINER_ID
```

### For `tf2-zoo` models
```
time ck run cmdgen:benchmark.mlperf-inference-vision --verbose --sut=chai --model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tf2-zoo --separator=:):$(ck list_variations misc --query_module_uoa=package --tags=model,tf1-zoo --separator=:) --library=tensorflow-v2.7.1-gpu --device_ids=0 --scenario=singlestream --group.open --dataset_size=5000  --batch_size=1 --target_latency_file=/home/krai/CK_REPOS/ck-mlperf/program/mlperf-inference-vision/target_latency.chai.txt --container=$CONTAINER_ID
```

### Running One Model

Example on running `ssd_mobilenet_v1_coco` from `tf1-zoo`

#### Accuracy
```
time ck run cmdgen:benchmark.mlperf-inference-vision --verbose --sut=chai --model=ssd_mobilenet_v1_coco --library=tensorflow-v2.7.1-gpu --device_ids=0 --scenario=singlestream --group.open --dataset_size=5000  --batch_size=1 --container=$CONTAINER_ID --mode=accuracy
```

#### Performance
```
time ck run cmdgen:benchmark.mlperf-inference-vision --verbose --sut=chai --model=ssd_mobilenet_v1_coco --library=tensorflow-v2.7.1-gpu --device_ids=0 --scenario=singlestream --group.open --dataset_size=5000  --batch_size=1 --target_latency_file=/home/krai/CK_REPOS/ck-mlperf/program/mlperf-inference-vision/target_latency.chai.txt --container=$CONTAINER_ID --mode=performance
```

<!-- #### Compliance
```
time ck run cmdgen:benchmark.mlperf-inference-vision --verbose --sut=chai --model=ssd_mobilenet_v1_coco --library=tensorflow-v2.7.1-gpu --device_ids=0 --scenario=singlestream --group.open --dataset_size=5000  --batch_size=1 --target_latency_file=/home/krai/CK_REPOS/ck-mlperf/program/mlperf-inference-vision/target_latency.chai.txt --container=$CONTAINER_ID --compliance,=TEST04-A,TEST04-B,TEST05,TEST01
``` -->