# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X3(A30x1):
    system = KnownSystem.A30x3
    gpu_batch_size = 1792
    server_target_qps = 9500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM(A30x1):
    system = KnownSystem.A30x8_Custom
    gpu_batch_size = 1792
    server_target_qps = 35200
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100X10_TRT(ServerGPUBaseConfig):
    system = KnownSystem.A100x10_TRT
    gpu_batch_size = 2048
    server_target_qps = 120000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2X4_TRT(ServerGPUBaseConfig):
    system = KnownSystem.A2x4_TRT
    audio_buffer_num_lines = 512
    dali_pipeline_depth = 1
    gpu_copy_streams = 4
    num_warmups = 32
    gpu_batch_size = 256
    audio_batch_size = 32
    server_target_qps = 2750
