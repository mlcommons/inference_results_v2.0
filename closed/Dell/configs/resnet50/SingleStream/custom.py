# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR12_edge_A2x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XR12_edge_A2x1
    single_stream_expected_latency_ns = 730000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A30x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XE2420_A30x1
    single_stream_expected_latency_ns = 660000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE2420_A30x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XE2420_A30x1
    single_stream_expected_latency_ns = 660000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_1XT4_EDGE(SingleStreamGPUBaseConfig):
    system = KnownSystem.XE2420_1xT4_edge
    single_stream_expected_latency_ns = 996648


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GB_1XMIG(SingleStreamGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GB_1xMIG

    start_from_device = True
    single_stream_expected_latency_ns = 720000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBX1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GBx1

    start_from_device = True
    single_stream_expected_latency_ns = 660000


