# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X3(A30x1):
    system = KnownSystem.A30x3
    offline_expected_qps = 6400
    gpu_batch_size = 1024
    gpu_copy_streams = 4
    gpu_inference_streams = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X3_HighAccuracy(A30X3):
    precision = "fp16"
    offline_expected_qps = A30X3.offline_expected_qps / 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X1_MAXQ(A30x1):
    system = KnownSystem.A30x1_MaxQ


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM(A30x1):
    system = KnownSystem.A30x8_Custom
    offline_expected_qps = 14000
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X8_CUSTOM_HighAccuracy(A30X8_CUSTOM):
    precision = "fp16"
    offline_expected_qps = A30X8_CUSTOM.offline_expected_qps / 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X1_CUSTOM(A30x1):
    system = KnownSystem.A30x1_Custom
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 1971.9999999999998
    workspace_size = 7516192768
    start_from_device = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100X10_TRT(OfflineGPUBaseConfig):
    system = KnownSystem.A100x10_TRT
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 35000
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100X10_TRT_HighAccuracy(A100X10_TRT):
    precision = "fp16"
    offline_expected_qps = 19000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2X4_TRT(OfflineGPUBaseConfig):
    system = KnownSystem.A2x4_TRT
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 256
    offline_expected_qps = 1200


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A2X4_TRT_HighAccuracy(A2X4_TRT):
    precision = "fp16"
    gpu_inference_streams = 1
    offline_expected_qps = 600
