# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X3(A30x1):
    system = KnownSystem.A30x3
    gpu_batch_size = 1800
    gpu_copy_streams = 4
    gpu_inference_streams = 1
    offline_expected_qps = 50000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X1_MAXQ(A30x1):
    system = KnownSystem.A30x1_MaxQ


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM(A30x1):
    system = KnownSystem.A30x8_Custom
    gpu_batch_size = 2048
    gpu_copy_streams = 4
    offline_expected_qps = 156000
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X1_CUSTOM(A30x1):
    system = KnownSystem.A30x1_Custom
    gpu_batch_size = 1536
    run_infer_on_copy_streams = True
    offline_expected_qps = 19278.375
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100X10_TRT(OfflineGPUBaseConfig):
    system = KnownSystem.A100x10_TRT
    gpu_batch_size = 2048
    offline_expected_qps = 390000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2X4_TRT(OfflineGPUBaseConfig):
    system = KnownSystem.A2x4_TRT
    gpu_batch_size = 1024
    offline_expected_qps = 25000
    gpu_inference_streams = 2
    gpu_copy_streams = 4
    run_infer_on_copy_streams = None
