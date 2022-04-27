# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X3(A30x1):
    system = KnownSystem.A30x3
    server_target_qps = 43000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM(A30x1):
    system = KnownSystem.A30x8_Custom
    server_target_qps = 134000
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100X10_TRT(ServerGPUBaseConfig):
    system = KnownSystem.A100x10_TRT
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 5
    server_target_qps = 325000
    use_cuda_thread_per_device = True
    use_graphs = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2X4_TRT(ServerGPUBaseConfig):
    system = KnownSystem.A2x4_TRT
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 4
    server_target_qps = 9210
    use_cuda_thread_per_device = True
    use_graphs = True
