# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR12_A2x1(ServerGPUBaseConfig):
    system = KnownSystem.XR12_A2x1
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 4
    server_target_qps = 4691//2
    use_cuda_thread_per_device = True
    use_graphs = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR12_A2x1_MaxQ(ServerGPUBaseConfig):
    system = KnownSystem.XR12_A2x1
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 4
    server_target_qps = 4691//2
    use_cuda_thread_per_device = True
    use_graphs = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class DSS8440_A100_PCIE_80GBx10(ServerGPUBaseConfig):
    system = KnownSystem.DSS8440_A100_PCIE_80GBx10
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 260000
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4

    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 256
    gpu_copy_streams = 4
    gpu_inference_streams = 5
    #server_target_qps = 135000
    server_target_qps = 135700
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBx4_Triton(R750XA_A100_PCIE_80GBx4):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A30x2(ServerGPUBaseConfig):
    system = KnownSystem.XE2420_A30x2
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 33180
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_T4X1(ServerGPUBaseConfig):
    system = KnownSystem.XE2420_T4x1
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 26
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 4650
    use_cuda_thread_per_device = False
    use_graphs = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_T4X1_Triton(XE2420_T4X1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GBx4

    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    precision = "int8"
    use_deque_limit = True
    deque_timeout_usec = 4000
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 139500
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True
    workspace_size: 7000000000
    numa_config: "0-15,128-143&1:16-31,144-159&32-47,160-175&0:48-63,176-191&64-79,192-207&3:80-95,208-223&96-111,224-239&2:112-127,240-255&112-127,240-255"
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBX4_Triton(XE8545_A100_SXM_80GBX4):
    use_triton = True

    gpu_copy_streams = 1
    gpu_inference_streams = 2
    instance_group_count = 2
    batch_triton_requests = False
    gather_kernel_buffer_threshold = 0
    num_concurrent_batchers = 8
    num_concurrent_issuers = 8
    request_timeout_usec = 2000
    server_target_qps = 122500


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE8545_MAXQ_A100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.XE8545_maxQ_A100_SXM_80GBx4

    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 4000
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 117000
    start_from_device = True
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    power_limit = 240
    numa_config: "0-15,128-143&1:16-31,144-159&32-47,160-175&0:48-63,176-191&64-79,192-207&3:80-95,208-223&96-111,224-239&2:112-127,240-255&112-127,240-255"


