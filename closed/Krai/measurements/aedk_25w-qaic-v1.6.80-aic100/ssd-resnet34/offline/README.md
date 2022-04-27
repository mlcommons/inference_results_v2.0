# Setup
Set up your system as detailed [here](https://github.com/krai/ck-qaic/blob/main/script/setup.aedk/README.md).

# Benchmarking
```
SDK_VER=v1.6.80 POWER=yes SUT=aedk_25w DOCKER=no OFFLINE_ONLY=yes WORKLOADS="ssd_resnet34" $(ck find ck-qaic:script:run)/run_edge.sh
```
