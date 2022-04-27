# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py


import os
import sys
sys.path.insert(0, os.getcwd())

from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *
from configs.rnnt import GPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    use_graphs = True
    gpu_inference_streams = 1
    gpu_copy_streams = 1
    num_warmups = 20480
    nobatch_sorting = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_4_R4900G5(ServerGPUBaseConfig):
    system = KnownSystem.A30_4_R4900G5
    gpu_batch_size = 1792
    server_target_qps = 10500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5(ServerGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_R5300G5
    gpu_batch_size = 2048         
    server_target_qps =36500         
    gpu_inference_streams = 1                 
    gpu_copy_streams = 4  
    dali_pipeline_depth = 1     
    server_num_issue_query_threads = 0
