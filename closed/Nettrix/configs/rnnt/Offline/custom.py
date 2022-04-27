# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X3(A30x1):
    system = KnownSystem.A30x3
    offline_expected_qps = 20880
    gpu_batch_size = 2048
    gpu_copy_streams = 4
    gpu_inference_streams = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X1_MAXQ(A30x1):
    system = KnownSystem.A30x1_MaxQ


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM(A30x1):
    system = KnownSystem.A30x8_Custom
    offline_expected_qps = 57000
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X1_CUSTOM(A30x1):
    system = KnownSystem.A30x1_Custom
    gpu_batch_size = 2048
    offline_expected_qps = 6959.999999999999
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100X10_TRT(OfflineGPUBaseConfig):
    system = KnownSystem.A100x10_TRT
    gpu_batch_size = 2048
    offline_expected_qps = 140000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2X4_TRT(OfflineGPUBaseConfig):
    system = KnownSystem.A2x4_TRT
    dali_batches_issue_ahead = 2
    dali_pipeline_depth = 2
    gpu_batch_size = 512
    audio_batch_size = 128
    offline_expected_qps = 4500
