# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR12_A2x1(ServerGPUBaseConfig):
    system = KnownSystem.XR12_A2x1
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 4
    server_target_qps = 16

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR12_A2x1_MaxQ(ServerGPUBaseConfig):
    system = KnownSystem.XR12_A2x1
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 4
    server_target_qps = 16

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class DSS8440_A100_PCIE_80GBx10(ServerGPUBaseConfig):
    system = KnownSystem.DSS8440_A100_PCIE_80GBx10
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 38041
    gpu_batch_size = 34
    gpu_inference_streams = 1
    server_target_qps = 8000
    active_sms = 100
    use_graphs = False
    use_cuda_thread_per_device = True
    start_from_device = True
    input_dtype = "int8"
    input_format = "linear"
    precision = "int8"


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps = 3250
    start_from_device = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBx4_Triton(R750XA_A100_PCIE_80GBx4):
    use_triton = True
    server_target_qps = 2850


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A30x2(ServerGPUBaseConfig):
    system = KnownSystem.XE2420_A30x2
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps = 913


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_T4X1(ServerGPUBaseConfig):
    system = KnownSystem.XE2420_T4x1
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 2
    gpu_inference_streams = 4
    server_target_qps = 100
    use_cuda_thread_per_device = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_T4X1_Triton(XE2420_T4X1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GBx4

    active_sms = 100
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 4
    server_target_qps = 3900
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    workspace_size: 7000000000
    start_from_device = True
    numa_config: "0-15,128-143&1:16-31,144-159&32-47,160-175&0:48-63,176-191&64-79,192-207&3:80-95,208-223&96-111,224-239&2:112-127,240-255&112-127,240-255"


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBX4_Triton(XE8545_A100_SXM_80GBX4):
    use_triton = True

    instance_group_count = 4
    deque_timeout_usec = 32205
    gpu_batch_size = 9
    gpu_inference_streams = 9
    server_target_qps = 3750


