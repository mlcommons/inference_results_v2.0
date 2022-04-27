# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5(ServerGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_R5300G5
    use_graphs = True
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 12150
    soft_drop = 0.993


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5_HighAccuracy(A30X8_CUSTOM_R5300G5):
    precision = "fp16"
    server_target_qps = 5390    
    use_graphs = True
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    soft_drop = 0.993


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5_Triton(A30X8_CUSTOM_R5300G5):
    use_triton = True
    use_graphs = False       
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 12150
    soft_drop = 0.993

	


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5_HighAccuracy_Triton(A30X8_CUSTOM_R5300G5_HighAccuracy):
    use_triton = True
    precision = "fp16"
    server_target_qps = 5770
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    soft_drop = 0.993




