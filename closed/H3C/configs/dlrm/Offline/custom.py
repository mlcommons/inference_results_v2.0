# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5(OfflineGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_R5300G5
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18         
    gpu_batch_size = 262100                   
    offline_expected_qps =1220000           
    max_pairs_per_staging_thread = 262100
    num_staging_batches =4         
    num_staging_threads =4         
    use_jemalloc = True
    numa_config = "0-3:0-39&4-7:40-79"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5_HighAccuracy(A30X8_CUSTOM_R5300G5):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5_Triton(A30X8_CUSTOM_R5300G5):
    use_triton = True
    batch_triton_requests = True
    num_concurrent_batchers = 1
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    offline_expected_qps = 1020000                
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5_HighAccuracy_Triton(A30X8_CUSTOM_R5300G5_HighAccuracy):
    use_triton = True
    batch_triton_requests = True
    num_concurrent_batchers = 1
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    offline_expected_qps =1220000               
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True
