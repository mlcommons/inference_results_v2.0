# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X3(A30x1):
    system = KnownSystem.A30x3
    offline_expected_qps = 1410

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X1_MAXQ(A30x1):
    system = KnownSystem.A30x1_MaxQ


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM(A30x1):
    system = KnownSystem.A30x8_Custom
    offline_expected_qps = 4500
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X1_CUSTOM(OfflineGPUBaseConfig):
    system = KnownSystem.A30x1_Custom
    gpu_batch_size = 32
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    offline_expected_qps = 483
    run_infer_on_copy_streams = False
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100X10_TRT(OfflineGPUBaseConfig):
    system = KnownSystem.A100x10_TRT
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    offline_expected_qps = 9500
    run_infer_on_copy_streams = False


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2X4_TRT(OfflineGPUBaseConfig):
    system = KnownSystem.A2x4_TRT
    gpu_batch_size = 32
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 300
