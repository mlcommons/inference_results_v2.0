# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2460M1_A30X4(ServerGPUBaseConfig):
    system = KnownSystem.GX2460M1_A30x4
    gpu_batch_size = 1792
    #server_target_qps = 18200
    server_target_qps = 17700
    numa_config = "0:12-15,44-47&1:8-11,40-43&2:28-31,60-63&3:20-23,52-55"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2460M1_A30X2(ServerGPUBaseConfig):
    system = KnownSystem.GX2460M1_A30x2

    gpu_batch_size = 1792
    server_target_qps = 9100
    numa_config = "0:12-15,44-47&1:8-11,40-43"


