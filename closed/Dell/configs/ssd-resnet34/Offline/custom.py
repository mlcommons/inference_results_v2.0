# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR12_A2x1(OfflineGPUBaseConfig):
    system = KnownSystem.XR12_A2x1
    gpu_batch_size = 32
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 69

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR12_A2x1_MaxQ(OfflineGPUBaseConfig):
    system = KnownSystem.XR12_A2x1
    gpu_batch_size = 32
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 69

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class DSS8440_A100_PCIE_80GBx10(OfflineGPUBaseConfig):
    system = KnownSystem.DSS8440_A100_PCIE_80GBx10
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    offline_expected_qps = 9600
    run_infer_on_copy_streams = False


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    offline_expected_qps = 3840.0
    run_infer_on_copy_streams = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBx4_Triton(R750XA_A100_PCIE_80GBx4):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A30x1(OfflineGPUBaseConfig):
    system = KnownSystem.XE2420_A30x1 
    gpu_batch_size = 32
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    offline_expected_qps = 470
    run_infer_on_copy_streams = False


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A30x2(OfflineGPUBaseConfig):
    system = KnownSystem.XE2420_A30x2
    gpu_batch_size = 32
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    offline_expected_qps = 1060
    run_infer_on_copy_streams = False


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_T4X1(OfflineGPUBaseConfig):
    system = KnownSystem.XE2420_T4x1
    gpu_batch_size = 12
    gpu_copy_streams = 4
    gpu_inference_streams = 1
    offline_expected_qps = 140


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_T4X1_Triton(XE2420_T4X1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GB_1XMIG(OfflineGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GB_1xMIG

    gpu_batch_size = 32
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    start_from_device = True
    offline_expected_qps = 135


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GBx4

    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    run_infer_on_copy_streams = False
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 4500
    workspace_size: 7000000000
    start_from_device = True
    numa_config: "0-15,128-143&1:16-31,144-159&32-47,160-175&0:48-63,176-191&64-79,192-207&3:80-95,208-223&96-111,224-239&2:112-127,240-255&112-127,240-255"


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBX4_Triton(XE8545_A100_SXM_80GBX4):
    use_triton = True


