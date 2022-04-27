# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5(ServerGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_R5300G5
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 132400
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5_Triton(A30X8_CUSTOM_R5300G5):
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 116000               
    use_cuda_thread_per_device = True
    use_graphs = False
    use_triton = True
    gather_kernel_buffer_threshold = 64
    max_queue_delay_usec = 1000
    request_timeout_usec = 8000


