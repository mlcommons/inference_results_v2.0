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


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC24ads_A100_v4(ServerGPUBaseConfig):
    system = KnownSystem.NC24ads_A100_v4
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 26000
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC24ads_A100_v4_Triton(NC24ads_A100_v4):
    server_target_qps = 24000
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC48ads_A100_v4(ServerGPUBaseConfig):
    system = KnownSystem.NC48ads_A100_v4
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 59000
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC48ads_A100_v4_Triton(NC48ads_A100_v4):
    server_target_qps = 45000
    use_triton = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4(ServerGPUBaseConfig):
    system = KnownSystem.NC96ads_A100_v4
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 119600
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4_Triton(NC96ads_A100_v4):
    server_target_qps = 90000
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(ServerGPUBaseConfig):
    system = KnownSystem.A30_MIG_1x1g_6gb
    use_deque_limit = True
    deque_timeout_usec = 200
    gpu_batch_size = 8
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 3400
    use_cuda_thread_per_device = True
    use_graphs = True
    workspace_size = 1610612736


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    server_target_qps = 3100


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(A30_MIG_1x1g6gb):
    server_target_qps = 1800
    gpu_inference_streams = 3
    use_graphs = False
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_32x1g6gb_Triton(ServerGPUBaseConfig):
    system = KnownSystem.A30_MIG_32x1g_6gb
    use_deque_limit = True
    deque_timeout_usec = 1000
    gpu_batch_size = 8
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    server_target_qps = 65000
    workspace_size = 1610612736
    use_graphs = False
    use_triton = True
    instance_group_count = 1
    max_queue_delay_usec = 1000
    preferred_batch_size = 8


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(ServerGPUBaseConfig):
    system = KnownSystem.A30x1
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 15079.999999999998
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_Triton(A30x1):
    server_target_qps = 13919.999999999998
    use_graphs = False
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8(A30x1):
    system = KnownSystem.A30x8
    server_target_qps = 116000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8_Triton(A30x8):
    server_target_qps = 110000
    use_graphs = False
    use_triton = True
    gather_kernel_buffer_threshold = 64
    max_queue_delay_usec = 1000
    request_timeout_usec = 8000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96amsr_A100_v4(ServerGPUBaseConfig):
    system = KnownSystem.ND96amsr_A100_v4
    use_deque_limit = True
    deque_timeout_usec = 4000
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 260000
    start_from_device = True
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96amsr_A100_v4_Triton(ND96amsr_A100_v4):
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    instance_group_count = 2
    batch_triton_requests = False
    server_target_qps = 200000
    gather_kernel_buffer_threshold = 0
    num_concurrent_batchers = 8
    num_concurrent_issuers = 8
    request_timeout_usec = 2000
    start_from_device = False
    use_triton = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96asr_v4(ServerGPUBaseConfig):
    system = KnownSystem.ND96asr_v4
    use_deque_limit = True
    gpu_copy_streams = 4
    start_from_device = True
    use_cuda_thread_per_device = True
    use_graphs = True
    deque_timeout_usec = 4000
    gpu_batch_size = 128
    gpu_inference_streams = 2
    server_target_qps = 256000

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96asr_v4_Triton(ND96asr_v4):
    server_target_qps = 160000
    start_from_device = False
    use_graphs = False
    use_triton = True
