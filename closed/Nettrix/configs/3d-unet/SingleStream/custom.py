# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X1_MAXQ(A30x1):
    system = KnownSystem.A30x1_MaxQ


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X1_MAXQ_HighAccuracy(A30X1_MAXQ):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X1_CUSTOM(A30x1):
    system = KnownSystem.A30x1_Custom
    gpu_batch_size = 1
    single_stream_expected_latency_ns = 1226400000
    start_from_device = True
    end_on_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X1_CUSTOM_HighAccuracy(A30X1_CUSTOM):
    pass
