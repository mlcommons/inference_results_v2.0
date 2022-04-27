# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR12_edge_A2x1(OfflineGPUBaseConfig):
    system = KnownSystem.XR12_edge_A2x1
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 256
    offline_expected_qps = 250

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
Class XR12_A2x1(OfflineGPUBaseConfig):
    system = KnownSystem.XR12_A2x1
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 256
    offline_expected_qps = 250


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
Class XR12_A2x1_HighAccuracy(XR12_A2x1):
    precision = "fp16"
    gpu_inference_streams = 1
    offline_expected_qps = 120

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
Class XR12_A2x1_MaxQ(OfflineGPUBaseConfig):
    system = KnownSystem.XR12_A2x1
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 256
    offline_expected_qps = 250

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class DSS8440_A100_PCIE_80GBx10(OfflineGPUBaseConfig):
    system = KnownSystem.DSS8440_A100_PCIE_80GBx10
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 34000
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class DSS8440_A100_PCIE_80GBx10_HighAccuracy(DSS8440_A100_PCIE_80GBx10):
    precision = "fp16"
    offline_expected_qps = 17500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 15000
    workspace_size = 7516192768 


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBx4_HighAccuracy(R750XA_A100_PCIE_80GBx4):
    precision = "fp16"
    #offline_expected_qps = R750XA_A100_PCIE_80GBx4.offline_expected_qps / 2
    offline_expected_qps = 7000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBx4_Triton(R750XA_A100_PCIE_80GBx4):
    use_triton = True
    offline_expected_qps = 13000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBx4_HighAccuracy_Triton(R750XA_A100_PCIE_80GBx4_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A30x1(OfflineGPUBaseConfig):
    system = KnownSystem.XE2420_A30x1
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 1971.9999999999998
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE2420_A30x1(OfflineGPUBaseConfig):
    system = KnownSystem.XE2420_A30x1
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 1971.9999999999998
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A30x2(OfflineGPUBaseConfig):
    system = KnownSystem.XE2420_A30x2
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 7000
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE2420_A30x2_HighAccuracy(XE2420_A30x2):
    precision = "fp16"
    offline_expected_qps = 5000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_T4X1(OfflineGPUBaseConfig):
    system = KnownSystem.XE2420_T4x1
    enable_interleaved = True
    use_small_tile_gemm_plugin = False
    gpu_batch_size = 256
    offline_expected_qps = 430

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE2420_T4X1_HighAccuracy(XE2420_T4X1):
    precision = "fp16"
    gpu_inference_streams = 1
    offline_expected_qps = 210


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_T4X1_Triton(XE2420_T4X1):
    use_triton = True

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE2420_T4X1_HighAccuracy_Triton(XE2420_T4X1_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GBx4

    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 20000
    workspace_size = 7516192768
    start_from_device = True
    numa_config: "0-15,128-143&1:16-31,144-159&32-47,160-175&0:48-63,176-191&64-79,192-207&3:80-95,208-223&96-111,224-239&2:112-127,240-255&112-127,240-255"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBX4_HighAccuracy(XE8545_A100_SXM_80GBX4):
    precision = "fp16"
    offline_expected_qps = XE8545_A100_SXM_80GBX4.offline_expected_qps / 2
    gpu_batch_size = 512


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GB_1XMIG(OfflineGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GB_1xMIG

    start_from_device = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 64
    offline_expected_qps = 500
    workspace_size = 2147483648


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBX1(OfflineGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GBx1

    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1280
    start_from_device = True
    gpu_inference_streams = 1
    offline_expected_qps = 5000

