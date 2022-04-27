# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X3(A30x1):
    system = KnownSystem.A30x3
    gpu_batch_size = 131000
    num_staging_batches = 8
    num_staging_threads = 8
    server_target_qps = 240000
    use_jemalloc = False
    numa_config = "0:0-15,32-47&1-2:16-31,48-63"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X3_HighAccuracy(A30X3):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM(A30x1):
    system = KnownSystem.A30x8_Custom
    gpu_batch_size = 131000
    num_staging_batches = 8
    num_staging_threads = 8
    server_target_qps = 950100
    use_jemalloc = False
    numa_config = "0-3:0-39,80-119&4-7:40-79,120-159"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X8_CUSTOM_HighAccuracy(A30X8_CUSTOM):
    server_target_qps = 949000
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100X10_TRT(ServerGPUBaseConfig):
    system = KnownSystem.A100x10_TRT
    deque_timeout_usec = 1
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 1300000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18
    numa_config = "0-4:0-39,80-119&5-9:40-79,120-159"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100X10_TRT_HighAccuracy(A100X10_TRT):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2X4_TRT(ServerGPUBaseConfig):
    system = KnownSystem.A2x4_TRT
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.5
    gemm_plugin_fairshare_cache_size = 18
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    gpu_batch_size = 100000
    server_target_qps = 84000
    use_jemalloc = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A2X4_TRT_HighAccuracy(A2X4_TRT):
    pass
