# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X3(A30x1):
    system = KnownSystem.A30x3
    offline_expected_qps = 4.92
    numa_config = "0:0-15,32-47&1-2:16-31,48-63"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X3_HighAccuracy(A30X3):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X1_MAXQ(A30x1):
    system = KnownSystem.A30x1_MaxQ


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X1_MAXQ_HighAccuracy(A30X1_MAXQ):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM(A30x1):
    system = KnownSystem.A30x8_Custom
    offline_expected_qps = 14
    start_from_device = True
    end_on_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X8_CUSTOM_HighAccuracy(A30X8_CUSTOM):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X1_CUSTOM(A30x1):
    system = KnownSystem.A30x1_Custom
    gpu_batch_size = 1
    offline_expected_qps = 1.69
    start_from_device = True
    end_on_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X1_CUSTOM_HighAccuracy(A30X1_CUSTOM):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100X10_TRT(OfflineGPUBaseConfig):
    system = KnownSystem.A100x10_TRT
    gpu_batch_size = 1
    offline_expected_qps = 30
    numa_config = "0-4:0-39,80-119&5-9:40-79,120-159"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100X10_TRT_HighAccuracy(A100X10_TRT):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2X4_TRT(OfflineGPUBaseConfig):
    system = KnownSystem.A2x4_TRT
    gpu_batch_size = 1
    offline_expected_qps = 2.0


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A2X4_TRT_HighAccuracy(A2X4_TRT):
    pass
