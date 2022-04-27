# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4
    deque_timeout_usec = 1
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 1130000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18
    start_from_device = True
    numa_config = "0-1:0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126&2-3:1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107,109,111,113,115,117,119,121,123,125,127"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBx4_HighAccuracy(R750XA_A100_PCIE_80GBx4):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBx4_Triton(R750XA_A100_PCIE_80GBx4):
    use_triton = True
    server_target_qps = 500000
    batch_triton_requests = True
    buffer_manager_thread_count = 0
    max_queue_delay_usec = 10000
    gather_kernel_buffer_threshold = 64


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBx4_HighAccuracy_Triton(R750XA_A100_PCIE_80GBx4_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_T4X1(ServerGPUBaseConfig):
    system = KnownSystem.XE2420_T4x1
    enable_interleaved_top_mlp = True
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.5
    gpu_batch_size = 65500
    gpu_num_bundles = 2
    num_staging_batches = 2
    num_staging_threads = 4
    server_target_qps = 24000
    use_jemalloc = True




@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE2420_T4X1_HighAccuracy(XE2420_T4X1):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_T4X1_Triton(XE2420_T4X1):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE2420_T4X1_HighAccuracy_Triton(XE2420_T4X1_HighAccuracy):
    use_triton = True


