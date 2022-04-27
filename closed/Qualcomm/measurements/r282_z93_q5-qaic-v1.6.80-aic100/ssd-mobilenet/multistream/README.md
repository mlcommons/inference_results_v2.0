# Initial System Setup 
Complete the system setup as detailed [here](https://github.com/krai/ck-qaic/blob/main/script/setup.docker/README.md)

# Benchmarking 
``` 
SDK_VER=v1.6.80 POWER=no SUT=r282_z93_q5 DOCKER=yes MULTISTREAM_ONLY=yes  WORKLOADS="ssd_mobilenet" $(ck find repo:ck-qaic)/scripts/benchmarking/run_edge.sh  
```