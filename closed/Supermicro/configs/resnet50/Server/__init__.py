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

from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *
from configs.resnet50 import GPUBaseConfig, CPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    active_sms = 100


class ServerCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Server


# AS-4124GO-NART_8_A100-SXM-80GB_TRT
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx8
    use_deque_limit = True
    deque_timeout_usec = 4000
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 285000
    start_from_device = True
    use_cuda_thread_per_device = True
    use_graphs = True

    numa_config = "0,1,2,3:0-31,64-95&4,5,6,7:32-63,96-127"


# SYS-220HE-FTNR_4_A2_TRT
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x4(ServerGPUBaseConfig):
    system = KnownSystem.A2x4
    use_deque_limit = True
    deque_timeout_usec = 2000
    # gpu_batch_size = 128
    gpu_batch_size = 16
    gpu_copy_streams = 4
    gpu_inference_streams = 4
    # server_target_qps = 4691
    server_target_qps = 10000
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS_2124GQ_NART(ServerGPUBaseConfig):
    _system_alias = "AS_2124GQ_NART_Redstone"
    _notes = "This is AS_2124GQ_NART"

    system = KnownSystem.AS_2124GQ_NART
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 4
    server_target_qps = 123000
    use_cuda_thread_per_device = True
    use_graphs = True
    numa_config = "1:0-31,128-159&0:32-63,160-191&3:64-95,192-223&2:96-127,224-255"


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS_2124GQ_NART_Triton(AS_2124GQ_NART):
    _system_alias = "AS_2124GQ_NART_Redstone"
    _notes = "This is AS_2124GQ_NART"

    deque_timeout_usec = 4000
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    instance_group_count = 2
    batch_triton_requests = False
    server_target_qps = 102380
    gather_kernel_buffer_threshold = 0
    num_concurrent_batchers = 8
    num_concurrent_issuers = 8
    request_timeout_usec = 2000
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class AS_2124GQ_NART_MaxQ(AS_2124GQ_NART):
    _system_alias = "AS_2124GQ_NART_Redstone"
    _notes = "This is AS_2124GQ_NART"

    power_limit = 225


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class AS_2124GQ_NART_Triton_MaxQ(AS_2124GQ_NART_MaxQ):
    _system_alias = "AS_2124GQ_NART_Redstone"
    _notes = "This is AS_2124GQ_NART"

    server_target_qps = 80000
    use_graphs = False
    use_triton = True



