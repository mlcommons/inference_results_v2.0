# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2460M1_A30X4(OfflineGPUBaseConfig):
    system = KnownSystem.GX2460M1_A30x4

    gpu_batch_size = 1536
    gpu_copy_streams = 3
    offline_expected_qps = 69000
    

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2460M1_A30X2(OfflineGPUBaseConfig):
    system = KnownSystem.GX2460M1_A30x2

    gpu_batch_size = 1536
    gpu_copy_streams = 3
    offline_expected_qps = 34500

