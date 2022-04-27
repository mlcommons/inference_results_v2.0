# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2460M1_A30X4(ServerGPUBaseConfig):
    system = KnownSystem.GX2460M1_A30x4
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 58000
    use_cuda_thread_per_device = True
    use_graphs = True

    

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2460M1_A30X2(ServerGPUBaseConfig):
    system = KnownSystem.GX2460M1_A30x2

    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 29000
    use_cuda_thread_per_device = True
    use_graphs = True


