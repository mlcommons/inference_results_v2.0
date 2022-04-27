# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A30x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XE2420_A30x1
    single_stream_expected_latency_ns = 340000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE2420_A30x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XE2420_A30x1
    single_stream_expected_latency_ns = 340000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GB_1XMIG(SingleStreamGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GB_1xMIG

    input_format = "chw4"
    start_from_device = True
    tensor_path = "build/preprocessed_data/coco/val2017/SSDMobileNet/int8_chw4"
    workspace_size = 1073741824
    single_stream_expected_latency_ns = 450000


