# MLPerf Inference v2.0 Implementations
This is a repository of Dell Technologies servers using optimized implementations for [MLPerf Inference Benchmark v2.0](https://www.mlperf.org/inference-overview/).

# GPU Implementations
## Benchmarks
**Please refer to /closed/NVIDIA for detailed instructions, including performace guides, and instructions on how to run with new systems.** 
  
The following benchmarks are part of our submission for MLPerf Inference v2.0:
- [3d-unet](code/3d-unet/tensorrt/README.md)
- [bert](code/bert/tensorrt/README.md)
- [dlrm](code/dlrm/tensorrt/README.md)
- [rnnt](code/rnnt/tensorrt/README.md)
- [resnet50](code/resnet50/tensorrt/README.md)
- [ssd-resnet34](code/ssd-resnet34/tensorrt/README.md)
- [ssd-mobilenet](code/ssd-mobilent/tensorrt/README.md)

# Dell Technologies Submission Systems

The closed systems that Dell has submitted on using NVIDIA GPUs are:
- Datacenter systems
  - Dell DSS 8440
    - A100-PCIe-80GB
  - Dell PowerEdge R750xa
    - A100-PCIe-80GB
  - Dell PowerEdge XE2420
    - A30
    - T4
  - Dell PowerEdge XE8545
    - A100-SXM-80GB / 500W
    - A100-SXM-80GB - 1x MIG 1g.10gb
  - Dell PowerEdge XR12
    - A2
- Edge systems
  - Dell PowerEdge XE2420
    - A30
    - T4
  - Dell PowerEdge XR12
    - A2

