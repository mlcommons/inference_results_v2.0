# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR12_edge_A2x1(OfflineGPUBaseConfig):
    system = KnownSystem.XR12_edge_A2x1
    dali_batches_issue_ahead = 2
    dali_pipeline_depth = 2
    gpu_batch_size = 512
    audio_batch_size = 128
    offline_expected_qps = 1000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR12_A2x1(OfflineGPUBaseConfig):
    system = KnownSystem.XR12_A2x1
    dali_batches_issue_ahead = 2
    dali_pipeline_depth = 2
    gpu_batch_size = 512
    audio_batch_size = 128
    offline_expected_qps = 1000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR12_A2x1_MaxQ(OfflineGPUBaseConfig):
    system = KnownSystem.XR12_A2x1
    dali_batches_issue_ahead = 2
    dali_pipeline_depth = 2
    gpu_batch_size = 512
    audio_batch_size = 128
    offline_expected_qps = 1000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class DSS8440_A100_PCIE_80GBx10(OfflineGPUBaseConfig):
    system = KnownSystem.DSS8440_A100_PCIE_80GBx10
    gpu_batch_size = 2048
    offline_expected_qps = 133000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4
    gpu_batch_size = 2048
    offline_expected_qps = 55000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A30x1(OfflineGPUBaseConfig):
    system = KnownSystem.XE2420_A30x1
    gpu_batch_size = 2048
    offline_expected_qps = 6959.999999999999


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE2420_A30x1(OfflineGPUBaseConfig):
    system = KnownSystem.XE2420_A30x1
    gpu_batch_size = 2048
    offline_expected_qps = 6959.999999999999


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_T4X1(OfflineGPUBaseConfig):
    system = KnownSystem.XE2420_T4x1
    audio_batch_size = 128
    disable_encoder_plugin = True
    gpu_batch_size = 2048
    gpu_copy_streams = 4
    offline_expected_qps = 1400
    num_warmups = 2048


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GBx4

    gpu_inference_streams = 1
    use_graphs = True
    audio_batch_size = 1024
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    offline_expected_qps = 55000
    num_warmups = 40480
    audio_buffer_num_lines = 4096
    nobatch_sorting = True
    scenario = Scenario.Offline
    benchmark = Benchmark.RNNT
    workspace_size: 7000000000
    numa_config: "0-15,128-143&1:16-31,144-159&32-47,160-175&0:48-63,176-191&64-79,192-207&3:80-95,208-223&96-111,224-239&2:112-127,240-255&112-127,240-255"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GB_1XMIG(OfflineGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GB_1xMIG

    audio_batch_size = 64
    audio_buffer_num_lines = 512
    dali_batches_issue_ahead = 2
    dali_pipeline_depth = 2
    gpu_batch_size = 1024
    offline_expected_qps = 1550
    num_warmups = 64
    workspace_size = 3221225472
    max_seq_length = 64


