# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


# AS-4124GO-NART_8_A100-SXM-80GB_TRT
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx8
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 16

    gpu_inference_streams = 2
    server_target_qps = 7720
    start_from_device = True

    numa_config = "0,1,2,3:0-31,64-95&4,5,6,7:32-63,96-127"


# SYS-220HE-FTNR_4_A2_TRT
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x4(ServerGPUBaseConfig):
    system = KnownSystem.A2x4
    gpu_copy_streams = 1
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 1
    gpu_inference_streams = 1
    server_target_qps = 220


