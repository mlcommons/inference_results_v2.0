# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X1_MAXQ(A30x1):
    system = KnownSystem.A30x1_MaxQ


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X1_CUSTOM(A30x1):
    system = KnownSystem.A30x1_Custom
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 1700000

