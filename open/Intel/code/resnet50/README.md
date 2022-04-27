# MLPerf Intel (Labs) OpenVino OMP CPP v2.0 Inference Build

This SW was tested on Ubuntu 18.04 and 20.04.

Benchmarks and Scenarios (dataset) in this release:
*  BootstrapNAS A: Offline and Server (imagenet only)
*  BootstrapNAS B: Offline and Server (imagenet only)
*  BootstrapNAS C: Offline and Server (imagenet only)

## Dataset
This benchmark uses the [ImageNet2012](http://image-net.org/challenges/LSVRC/2012/) validation set. To run the benchmarks, please provide the validation set (images including val_map.txt) as instructed in [How to Build and Run](#how-to-build-and-run).


## Bootstrap Models
BootstrapNAS models are provided in [src/models/](src/models)


## BKC on CPX, ICX systems
Use the following to optimize performance on CPX/ICX systems.  These BKCs are provided in `performance.sh` mentioned in [How to Build and Run](#how-to-build-and-run).
 - Turbo ON: ```echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo```
 - Set CPU governor to performance (Please rerun this command after reboot):  
	 - ```echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor``` 
           OR
	 - ```cpupower frequency-set -g performance```


## How to Build and Run
1. Navigate to [code/resnet50](.) directory. This directory is your BUILD_DIRECTORY.
2. Run the build script: ```./build-ovmlperf.sh```
3. Run the performance script for CPX/ICX systems: ```./performance.sh```
4. Provide imagenet dataset in ```src/imagenet```.
5. Navigate to ```src/scripts```
6. Modify BUILD_DIRECTORY in ```setup_env.sh```(if necessary) and source: ```source setup_env.sh```
7. Navigate to ```1-node-<sockets>-<platform>-ov``` to run desired test


### Performance
Syntax to run a **Performance** benchmark

```
./bootstrapnas-<scenario>.sh <model>
```

For instance, to run ```BootstrapNAS A``` Offline scenario:
```
./bootstrapnas-offline.sh A
```

### Accuracy
+ User first runs the benchmark in ```Accuracy``` mode to generate ```mlperf_log_accuracy.json```
+ User then runs a dedicated accuracy tool provided by MLPerf


Syntax to generate Accuracy logs:

```
./bootstrapnas-<scenario>-acc.sh <model>
```

For instance:

```
./bootstrapnas-offline-acc.sh A
```

To compute the **Top-1** accuracy for ```BootstrapNAS A``` (after running the Accuracy script), run the command below:
```
python </path/to/mlperf-inference>/vision/classification_and_detection/tools/accuracy-imagenet.py \
   --mlperf-accuracy-file mlperf_log_accuracy.json \
   --imagenet-val-file </path/to/dataset-imagenet-ilsvrc2012-val>/val_map.txt \
   --dtype float32
```

    
## Known issues

* Issue:
terminate called after throwing an instance of 'InferenceEngine::details::InferenceEngineException'
  what():  can't protect

Solution:
Patch with the following your current and any further submission machines as root:

 1. Add the following line to **/etc/sysctl.conf**: 
    vm.max_map_count=2097152 
 
 2. You may want to check that current value is
    too small with `cat /proc/sys/vm/max_map_count` 
    
 3. Reload the config as
    root: `sysctl -p` (this will print the updated value)
