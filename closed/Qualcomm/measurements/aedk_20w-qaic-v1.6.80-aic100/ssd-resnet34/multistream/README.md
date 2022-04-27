# Initial System Setup 
Complete the system setup as detailed [here](https://github.com/krai/ck-qaic/blob/main/script/setup.aedk/README.md)

# Benchmarking 
``` 
SDK_VER=v1.6.80 POWER=yes SUT=aedk_20w DOCKER=no MULTISTREAM_ONLY=yes  WORKLOADS="ssd_resnet34" $(ck find repo:ck-qaic)/scripts/benchmarking/run_edge.sh  
```