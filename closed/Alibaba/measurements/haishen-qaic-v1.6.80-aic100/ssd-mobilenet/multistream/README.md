# Setup
Set up your system as detailed [here](https://github.com/krai/ck-qaic/blob/main/script/setup.aedk/README.md).

# Benchmarking
```
SDK_VER=v1.6.80 POWER=no SUT=aedkh DOCKER=no MULTISTREAM_ONLY=yes WORKLOADS="ssd_mobilenet" $(ck find ck-qaic:script:run)/run_edge.sh
```
