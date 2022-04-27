# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X3(A30x1):
    system = KnownSystem.A30x3
    server_target_qps = 3800


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X3_HighAccuracy(A30X3):
    precision = "fp16"
    server_target_qps = 1700


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM(A30x1):
    system = KnownSystem.A30x8_Custom
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 11985
    soft_drop = 0.993
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X8_CUSTOM_HighAccuracy(A30X8_CUSTOM):
    precision = "fp16"
    server_target_qps = 5400


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100X10_TRT(ServerGPUBaseConfig):
    system = KnownSystem.A100x10_TRT
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 30500.0
    soft_drop = 1.0


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100X10_TRT_HighAccuracy(A100X10_TRT):
    precision = "fp16"
    server_target_qps = 14700


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2X4_TRT(ServerGPUBaseConfig):
    system = KnownSystem.A2x4_TRT
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 4
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    soft_drop = 0.993
    server_target_qps = 730


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A2X4_TRT_HighAccuracy(A2X4_TRT):
    precision = "fp16"
    gpu_batch_size = 8
    server_target_qps = 335
