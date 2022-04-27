# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR12_edge_A2x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XR12_edge_A2x1
    gpu_batch_size = 1
    single_stream_expected_latency_ns = 5378800000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A30x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XE2420_A30x1
    gpu_batch_size = 1
    single_stream_expected_latency_ns = 1226400000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE2420_A30x1_HighAccuracy(XE2420_A30x1):
    pass


