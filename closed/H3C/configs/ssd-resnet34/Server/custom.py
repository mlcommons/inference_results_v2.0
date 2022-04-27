# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py


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


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    active_sms = 100
    use_graphs = False
    use_cuda_thread_per_device = True


class ServerCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Server
    batch_size = 1
    use_deque_limit = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_4_R4900G5(ServerGPUBaseConfig):
    system = KnownSystem.A30_4_R4900G5
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps = 1786.4

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5(ServerGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_R5300G5
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps = 3720       


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5_Triton(A30X8_CUSTOM_R5300G5):
    active_sms = 100
    gpu_copy_streams = 4
    use_deque_limit = True
    use_graphs = False
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps = 3680            
    instance_group_count =2              
    use_triton = True
    max_queue_delay_usec = 500	
