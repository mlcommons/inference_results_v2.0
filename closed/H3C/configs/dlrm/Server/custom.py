# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5(ServerGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_R5300G5
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gpu_num_bundles = 2
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18      
    gpu_batch_size = 131000
    num_staging_batches = 8
    num_staging_threads = 8
    server_target_qps = 900000     
    use_jemalloc = False
    numa_config = "0,1,2,3:0-19&4,5,6,7:60-79"
 
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5_HighAccuracy(A30X8_CUSTOM_R5300G5):
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gpu_num_bundles = 2
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18      
    gpu_batch_size = 131000
    num_staging_batches = 8
    num_staging_threads = 8
    server_target_qps = 900000     
    use_jemalloc = False
    numa_config = "0-3:0-39&4-7:40-79"


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5_Triton(A30X8_CUSTOM_R5300G5):
    use_triton = True
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size =18     
    gpu_batch_size =131000          
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = False
    server_target_qps =880000         
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5_HighAccuracy_Triton(A30X8_CUSTOM_R5300G5_HighAccuracy):
    use_triton = True
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18        
    gpu_batch_size =131000          
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = False
    server_target_qps =880000         
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64


