# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR12_edge_A2x1(MultiStreamGPUBaseConfig):
    system = KnownSystem.XR12_edge_A2x1
    multi_stream_expected_latency_ns = 112350000
    # Use early stopping estimate due to long runtime.
    min_duration = 3600000
    min_query_count = 45000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A30x1(MultiStreamGPUBaseConfig):
    system = KnownSystem.XE2420_A30x1
    multi_stream_expected_latency_ns = 20100000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GB_1XMIG(MultiStreamGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GB_1xMIG

    start_from_device = True
    multi_stream_expected_latency_ns = 62000000
    # Use early stopping estimate due to long runtime.
    min_duration = 3600000
    min_query_count = 45000

