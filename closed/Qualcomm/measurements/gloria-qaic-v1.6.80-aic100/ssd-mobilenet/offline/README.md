# Initial System Setup 
Complete the system setup as detailed [here](https://github.com/krai/ck-qaic/blob/main/script/setup.aedk/README.md)

# Benchmarking 
``` 
SDK_VER=v1.6.80 POWER=no SUT=aedkg DOCKER=no OFFLINE_ONLY=yes  WORKLOADS="ssd_mobilenet" $(ck find repo:ck-qaic)/scripts/benchmarking/run_edge.sh  
```