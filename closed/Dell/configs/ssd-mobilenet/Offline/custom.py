# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A30x1(OfflineGPUBaseConfig):
    system = KnownSystem.XE2420_A30x1
    gpu_inference_streams = 2
    gpu_batch_size = 1024
    gpu_copy_streams = 4
    run_infer_on_copy_streams = False
    offline_expected_qps = 26000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE2420_A30x1(OfflineGPUBaseConfig):
    system = KnownSystem.XE2420_A30x1
    gpu_inference_streams = 2
    gpu_batch_size = 1024
    gpu_copy_streams = 4
    run_infer_on_copy_streams = False
    offline_expected_qps = 26000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GB_1XMIG(OfflineGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GB_1xMIG

    gpu_inference_streams = 1
    gpu_batch_size = 256
    gpu_copy_streams = 2
    start_from_device = True
    workspace_size = 2147483648
    offline_expected_qps = 7000


