# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2460M1_A30X4(OfflineGPUBaseConfig):
    system = KnownSystem.GX2460M1_A30x4
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    offline_expected_qps = 560000
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True
    numa_config = "0:12-15,44-47&1:8-11,40-43&2:28-31,60-63&3:20-23,52-55"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GX2460M1_A30X4_HighAccuracy(GX2460M1_A30X4):
    pass


# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2460M1_A30X2(OfflineGPUBaseConfig):
    system = KnownSystem.GX2460M1_A30x2
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    offline_expected_qps = 560000
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True
    numa_config = "0:12-15,44-47&1:8-11,40-43"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GX2460M1_A30X2_HighAccuracy(GX2460M1_A30X2):
    pass



