# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR12_edge_A2x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XR12_edge_A2x1
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 105000000
    nouse_copy_kernel = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A30x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XE2420_A30x1
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE2420_A30x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XE2420_A30x1
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GB_1XMIG(SingleStreamGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GB_1xMIG

    audio_buffer_num_lines = 4
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = True
    workspace_size = 1073741824


