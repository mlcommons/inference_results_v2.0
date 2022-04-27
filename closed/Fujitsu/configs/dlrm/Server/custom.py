# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2460M1_A30X4(ServerGPUBaseConfig):
    system = KnownSystem.GX2460M1_A30x4
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gpu_batch_size = 131000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 500000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18
    numa_config = "0:12-15,44-47&1:8-11,40-43&2:28-31,60-63&3:20-23,52-55"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GX2460M1_A30X4_HighAccuracy(GX2460M1_A30X4):
    pass



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2460M1_A30X2(ServerGPUBaseConfig):
    system = KnownSystem.GX2460M1_A30x2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gpu_batch_size = 226000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 165000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18
    numa_config = "0:12-15,44-47&1:8-11,40-43"



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GX2460M1_A30X2_HighAccuracy(GX2460M1_A30X2):
    pass


