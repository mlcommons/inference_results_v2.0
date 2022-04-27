# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X3(A30x1):
    system = KnownSystem.A30x3
    server_target_qps = 1150


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM(A30x1):
    system = KnownSystem.A30x8_Custom
    server_target_qps = 3800
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100X10_TRT(ServerGPUBaseConfig):
    system = KnownSystem.A100x10_TRT
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps = 9000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2X4_TRT(ServerGPUBaseConfig):
    system = KnownSystem.A2x4_TRT
    gpu_copy_streams = 2
    use_deque_limit = True
    deque_timeout_usec = 20000
    gpu_batch_size = 1
    gpu_inference_streams = 1
    server_target_qps = 250
    numa_config = "0,1:0-39,80-119&2,3:40-79,120-159"
