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


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC24ads_A100_v4(ServerGPUBaseConfig):
    system = KnownSystem.NC24ads_A100_v4
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps = 770

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC24ads_A100_v4_Triton(NC24ads_A100_v4):
    instance_group_count = 4
    use_triton = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC48ads_A100_v4(ServerGPUBaseConfig):
    system = KnownSystem.NC48ads_A100_v4
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps = 1500

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC48ads_A100_v4_Triton(NC48ads_A100_v4):
    server_target_qps = 1600 
    instance_group_count = 4
    use_triton = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4(ServerGPUBaseConfig):
    system = KnownSystem.NC96ads_A100_v4
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps = 3200

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4_Triton(NC96ads_A100_v4):
    server_target_qps = 3200
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(ServerGPUBaseConfig):
    system = KnownSystem.A30_MIG_1x1g_6gb
    gpu_copy_streams = 1
    use_deque_limit = True
    deque_timeout_usec = 50000
    gpu_batch_size = 2
    gpu_inference_streams = 1
    server_target_qps = 95
    use_cuda_thread_per_device = False
    workspace_size = 1610612736


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    server_target_qps = 85


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(A30_MIG_1x1g6gb):
    server_target_qps = 90
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_32x1g6gb_Triton(ServerGPUBaseConfig):
    system = KnownSystem.A30_MIG_32x1g_6gb
    gpu_copy_streams = 1
    use_deque_limit = True
    deque_timeout_usec = 50000
    gpu_batch_size = 2
    gpu_inference_streams = 1
    server_target_qps = 3000
    use_cuda_thread_per_device = False
    workspace_size = 1610612736
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(ServerGPUBaseConfig):
    system = KnownSystem.A30x1
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps = 446.59999999999997


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_Triton(A30x1):
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8(A30x1):
    system = KnownSystem.A30x8
    server_target_qps = 3572.7999999999997


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8_Triton(A30x8):
    server_target_qps = 3572
    instance_group_count = 4
    use_triton = True
    max_queue_delay_usec = 500


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A30x8_MaxQ(A30x8):
    server_target_qps = 3305.9999999999995
    power_limit = 200


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A30x8_Triton_MaxQ(A30x8_Triton):
    server_target_qps = 3248.0
    power_limit = 200
    max_queue_delay_usec = None

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96amsr_A100_v4(ServerGPUBaseConfig):
    system = KnownSystem.ND96amsr_A100_v4
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 4
    server_target_qps = 7580
    start_from_device = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96amsr_A100_v4_Triton(ND96amsr_A100_v4):
    start_from_device = None
    server_target_qps = 7100
    instance_group_count = 4
    use_triton = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96asr_v4(ServerGPUBaseConfig):
    system = KnownSystem.ND96asr_v4
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 4
    server_target_qps = 7650
    start_from_device = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96asr_v4_Triton(ND96asr_v4):
    start_from_device = None
    server_target_qps = 7100
    instance_group_count = 4
    use_triton = True
