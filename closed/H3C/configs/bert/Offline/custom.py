# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5(OfflineGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_R5300G5
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    gpu_copy_streams = 1
    offline_expected_qps =13500               
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5_HighAccuracy(A30X8_CUSTOM_R5300G5):
    precision = "fp16"
    offline_expected_qps =8000          
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5_Triton(A30X8_CUSTOM_R5300G5):
    use_triton = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 13000
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5_HighAccuracy_Triton(A30X8_CUSTOM_R5300G5_HighAccuracy):
    use_triton = True
    precision = "fp16"
    offline_expected_qps = 8000
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    workspace_size = 7516192768


