# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py


import os
import sys
sys.path.insert(0, os.getcwd())

from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *
from configs.rnnt import GPUBaseConfig


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    use_graphs = True
    num_warmups = 512
    audio_batch_size = 512
    audio_buffer_num_lines = 4096
    dali_batches_issue_ahead = 4
    dali_pipeline_depth = 4
    gpu_inference_streams = 1
    gpu_copy_streams = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_4_R4900G5(OfflineGPUBaseConfig):
    system = KnownSystem.A30_4_R4900G5
    gpu_batch_size = 2048
    offline_expected_qps = 27839

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5(OfflineGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_R5300G5
    gpu_batch_size = 2048
    offline_expected_qps =60000       
