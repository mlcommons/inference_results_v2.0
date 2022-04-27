# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR12_A2x1(ServerGPUBaseConfig):
    system = KnownSystem.XR12_A2x1
    audio_buffer_num_lines = 512
    dali_pipeline_depth = 1
    gpu_copy_streams = 4
    num_warmups = 32
    gpu_batch_size = 256
    audio_batch_size = 32
    server_target_qps = 1305//2

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR12_A2x1_MaxQ(ServerGPUBaseConfig):
    system = KnownSystem.XR12_A2x1
    audio_buffer_num_lines = 512
    dali_pipeline_depth = 1
    gpu_copy_streams = 4
    num_warmups = 32
    gpu_batch_size = 256
    audio_batch_size = 32
    server_target_qps = 1305//2

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class DSS8440_A100_PCIE_80GBx10(ServerGPUBaseConfig):
    system = KnownSystem.DSS8440_A100_PCIE_80GBx10
    gpu_batch_size = 1980
    server_target_qps = 116500
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    precision = "fp16"
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    gpu_copy_streams = 11
    nobatch_sorting = True
    num_warmups = 20480


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4
    gpu_batch_size = 2048
    server_target_qps = 48500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_T4X1(ServerGPUBaseConfig):
    system = KnownSystem.XE2420_T4x1
    audio_batch_size = 64
    audio_buffer_num_lines = 512
    audio_fp16_input = None
    dali_pipeline_depth = 1
    disable_encoder_plugin = True
    gpu_batch_size = 256
    gpu_copy_streams = 4
    max_seq_length = 102
    num_warmups = 2048
    server_target_qps = 950


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GBx4

    gpu_inference_streams = 1
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 20480
    server_num_issue_query_threads = 0
    server_target_qps = 53500
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT
    workspace_size: 7000000000
    numa_config: "0-15,128-143&1:16-31,144-159&32-47,160-175&0:48-63,176-191&64-79,192-207&3:80-95,208-223&96-111,224-239&2:112-127,240-255&112-127,240-255"


