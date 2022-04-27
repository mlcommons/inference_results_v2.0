# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR12_A2x1(ServerGPUBaseConfig):
    system = KnownSystem.XR12_A2x1
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    #active_sms = 10
    gpu_batch_size = 4
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 150
    soft_drop = 0.993

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR12_A2x1_MaxQ(ServerGPUBaseConfig):
    system = KnownSystem.XR12_A2x1
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    #active_sms = 10
    gpu_batch_size = 4
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 150
    soft_drop = 0.993

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XR12_A2x1_HighAccuracy(XR12_A2x1):
    precision = "fp16"
    gpu_batch_size = 8
    server_target_qps = 60

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class XR12_A2x1_HighAccuracy_MaxQ(XR12_A2x1):
    precision = "fp16"
    gpu_batch_size = 8
    server_target_qps = 60

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class DSS8440_A100_PCIE_80GBx10(ServerGPUBaseConfig):
    system = KnownSystem.DSS8440_A100_PCIE_80GBx10
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 29800
    soft_drop = 1.0
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = False
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    use_graphs = True
    gemm_plugin_fairshare_cache_size = 120
    use_small_tile_gemm_plugin = True
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class DSS8440_A100_PCIE_80GBx10_HighAccuracy(DSS8440_A100_PCIE_80GBx10):
    precision = "fp16"
    server_target_qps = 14150
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = False
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    input_dtype = "int32"
    input_format = "linear"
    use_graphs = True
    active_sms = 60
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 11700
    soft_drop = 1.0
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBx4_HighAccuracy(R750XA_A100_PCIE_80GBx4):
    precision = "fp16"
    #server_target_qps = R750XA_A100_PCIE_80GBx4.server_target_qps / 2
    server_target_qps = 5680


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBx4_Triton(R750XA_A100_PCIE_80GBx4):
    use_triton = True
    server_target_qps = 10700
    start_from_device = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBx4_HighAccuracy_Triton(R750XA_A100_PCIE_80GBx4_HighAccuracy):
    use_triton = True
    start_from_device = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A30x2(ServerGPUBaseConfig):
    system = KnownSystem.XE2420_A30x2
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 3085
    soft_drop = 0.993


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE2420_A30x2_HighAccuracy(XE2420_A30x2):
    precision = "fp16"
    server_target_qps = 1340


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_T4X1(ServerGPUBaseConfig):
    system = KnownSystem.XE2420_T4x1
    enable_interleaved = True
    active_sms = 100
    #gpu_batch_size = 16
    gpu_batch_size = 8
    graphs_max_seqlen = 260
    server_num_issue_query_threads = 0
    server_target_qps = 360
    soft_drop = 0.993
    gemm_plugin_fairshare_cache_size = None
    use_small_tile_gemm_plugin = None
    gpu_inference_streams = 1
    gpu_copy_streams = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE2420_T4X1_HighAccuracy(XE2420_T4X1):
    gpu_inference_streams = 1
    precision = "fp16"
    gpu_batch_size = 8
    server_target_qps = 160
    graph_specs = "(128, 4, 256, 4), (192, 128, 512, 4), (256, 192, 1536, 8), (384, 256, 2048, 16)"

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_T4X1_Triton(XE2420_T4X1):
    use_triton = True

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE2420_T4X1_HighAccuracy_Triton(XE2420_T4X1_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GBx4

    enable_interleaved = False
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    use_graphs = True
    active_sms = 60
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 48
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 13250
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    workspace_size: 7000000000
    start_from_device = True
    numa_config: "0-15,128-143&1:16-31,144-159&32-47,160-175&0:48-63,176-191&64-79,192-207&3:80-95,208-223&96-111,224-239&2:112-127,240-255&112-127,240-255"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBX4_HighAccuracy(XE8545_A100_SXM_80GBX4):
    precision = "fp16"
    server_target_qps = XE8545_A100_SXM_80GBX4.server_target_qps / 2
    gpu_batch_size = 24

