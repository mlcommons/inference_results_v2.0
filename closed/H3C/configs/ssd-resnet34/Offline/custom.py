# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

#

import os
import sys
sys.path.insert(0, os.getcwd())

from importlib import import_module
from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *


ParentConfig = import_module("configs.ssd-resnet34")
GPUBaseConfig = ParentConfig.GPUBaseConfig
CPUBaseConfig = ParentConfig.CPUBaseConfig


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    use_graphs = False


class OfflineCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Offline
    batch_size = 1
    run_infer_on_copy_streams = False



@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_4_R4900G5(OfflineGPUBaseConfig):
    system = KnownSystem.A30_4_R4900G5
    gpu_batch_size = 32
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    offline_expected_qps = 1880
    run_infer_on_copy_streams = False

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5(OfflineGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_R5300G5
    gpu_batch_size = 32
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    offline_expected_qps = 3900             
    run_infer_on_copy_streams = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5_Triton(A30X8_CUSTOM_R5300G5):
    gpu_batch_size = 32
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    offline_expected_qps =3860.0           
    run_infer_on_copy_streams = False
    use_triton = True