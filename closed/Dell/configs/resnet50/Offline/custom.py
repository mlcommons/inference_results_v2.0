# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR12_edge_A2x1(OfflineGPUBaseConfig):
    system = KnownSystem.XR12_edge_A2x1
    gpu_batch_size = 1024
    offline_expected_qps = 3022
    run_infer_on_copy_streams = None

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR12_A2x1(OfflineGPUBaseConfig):
    system = KnownSystem.XR12_A2x1
    gpu_batch_size = 1024
    offline_expected_qps = 3022
    run_infer_on_copy_streams = None

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR12_A2x1_MaxQ(OfflineGPUBaseConfig):
    system = KnownSystem.XR12_A2x1
    gpu_batch_size = 1024
    offline_expected_qps = 3022
    run_infer_on_copy_streams = None

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class DSS8440_A100_PCIE_80GBx10(OfflineGPUBaseConfig):
    system = KnownSystem.DSS8440_A100_PCIE_80GBx10
    gpu_batch_size = 2048
    offline_expected_qps = 370000
    gpu_inference_streams = 1
    use_graphs = False
    gpu_copy_streams = 2
    run_infer_on_copy_streams = False
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4
    gpu_batch_size = 2048
    offline_expected_qps = 147264


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBx4_Triton(R750XA_A100_PCIE_80GBx4):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE2420_A30x1(OfflineGPUBaseConfig):
    system = KnownSystem.XE2420_A30x1
    gpu_batch_size = 1536
    run_infer_on_copy_streams = True
    offline_expected_qps = 18200


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A30x1(OfflineGPUBaseConfig):
    system = KnownSystem.XE2420_A30x1
    gpu_batch_size = 1536
    run_infer_on_copy_streams = True
    offline_expected_qps = 18200


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE2420_A30x1(OfflineGPUBaseConfig):
    system = KnownSystem.XE2420_A30x1
    gpu_batch_size = 1536
    run_infer_on_copy_streams = True
    offline_expected_qps = 18200


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A30x2(OfflineGPUBaseConfig):
    system = KnownSystem.XE2420_A30x2
    gpu_batch_size = 1536
    run_infer_on_copy_streams = True
    offline_expected_qps = 35000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_T4X1(OfflineGPUBaseConfig):
    system = KnownSystem.XE2420_T4x1
    gpu_batch_size = 256
    gpu_copy_streams = 4
    offline_expected_qps = 6100


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_T4X1_Triton(XE2420_T4X1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_1XT4_EDGE(OfflineGPUBaseConfig):
    system = KnownSystem.XE2420_1xT4_edge
    gpu_batch_size = 256
    gpu_copy_streams = 4
    offline_expected_qps = 6100


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GB_1XMIG(OfflineGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GB_1xMIG

    gpu_batch_size = 256
    run_infer_on_copy_streams = True
    start_from_device = True
    offline_expected_qps = 5100


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBX1(OfflineGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GBx1

    gpu_batch_size = 2048
    gpu_copy_streams = 4
    run_infer_on_copy_streams = True
    start_from_device = True
    offline_expected_qps = 45000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GBx4

    gpu_inference_streams = 2
    use_graphs = False
    gpu_batch_size = 2048
    gpu_copy_streams = 3
    workspace_size = 7000000000
    start_from_device = True
    run_infer_on_copy_streams = False
    numa_config: "0-15,128-143&1:16-31,144-159&32-47,160-175&0:48-63,176-191&64-79,192-207&3:80-95,208-223&96-111,224-239&2:112-127,240-255&112-127,240-255"
    scenario = Scenario.Offline
    benchmark = Benchmark.ResNet50
    offline_expected_qps = 168400


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBX4_Triton(XE8545_A100_SXM_80GBX4):
    use_triton = True

    gpu_copy_streams = 1
    num_concurrent_batchers = 6
    num_concurrent_issuers = 6
    instance_group_count = 2
    batch_triton_requests = True
    offline_expected_qps = 160664


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GB_MIG_Triton(OfflineGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GB_MIG
    use_triton = True
    gpu_batch_size = 256
    run_infer_on_copy_streams = True
    offline_expected_qps = 180000
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE8545_MAXQ_A100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8545_maxQ_A100_SXM_80GBx4

    gpu_inference_streams = 2
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_graphs = False
    gpu_batch_size = 2048
    gpu_copy_streams = 3
    run_infer_on_copy_streams = False
    start_from_device = True
    scenario = Scenario.Offline
    benchmark = Benchmark.ResNet50
    offline_expected_qps = 171000
    power_limit = 240
    numa_config: "0-15,128-143&1:16-31,144-159&32-47,160-175&0:48-63,176-191&64-79,192-207&3:80-95,208-223&96-111,224-239&2:112-127,240-255&112-127,240-255"


