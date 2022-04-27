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
class A100_PCIex1(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_40GBx1
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 26000
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex1_Triton(A100_PCIex1):
    server_target_qps = 24000
    use_graphs = False
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex8(A100_PCIex1):
    system = KnownSystem.A100_PCIe_40GBx8
    gpu_batch_size = 128
    gpu_inference_streams = 5
    server_target_qps = 190000
    numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex8_Triton(A100_PCIex8):
    gpu_batch_size = 64
    deque_timeout_usec = 500
    gpu_inference_streams = 3
    server_target_qps = 175000
    numa_config = "3:0-7&2:8-15&1:16-23&0:24-31&7:32-39&6:40-47&5:48-55&4:56-63"
    use_graphs = False
    use_triton = True
    max_queue_delay_usec = 1000
    request_timeout_usec = 2000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBx1
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 26000
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1_Triton(A100_PCIe_80GBx1):
    server_target_qps = 24000
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBx8
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 160
    gpu_copy_streams = 4
    gpu_inference_streams = 5
    server_target_qps = 210000
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8_Triton(A100_PCIe_80GBx8):
    deque_timeout_usec = 500
    gpu_batch_size = 64
    gpu_inference_streams = 3
    server_target_qps = 181000
    numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"
    use_graphs = False
    use_triton = True
    max_queue_delay_usec = 1000
    request_timeout_usec = 2000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_MaxQ(A100_PCIe_80GBx8):
    numa_config = None
    gpu_batch_size = 64
    gpu_inference_streams = 3
    server_target_qps = 185000
    power_limit = 200


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_Triton_MaxQ(A100_PCIe_80GBx8_MaxQ):
    server_target_qps = 130000
    use_graphs = False
    power_limit = 175
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_ARMx1
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 26000
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1_Triton(A100_PCIe_80GB_aarch64x1):
    num_concurrent_batchers = 3
    num_concurrent_issuers = 3
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2(A100_PCIe_80GB_aarch64x1):
    system = KnownSystem.A100_PCIe_80GB_ARMx2
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 5
    server_target_qps = 52000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2_Triton(A100_PCIe_80GB_aarch64x1):
    system = KnownSystem.A100_PCIe_80GB_ARMx2
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 5
    num_concurrent_batchers = 4
    num_concurrent_issuers = 4
    server_target_qps = 52000
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4(A100_PCIe_80GB_aarch64x1):
    system = KnownSystem.A100_PCIe_80GB_ARMx4
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 5
    server_target_qps = 104000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4_Triton(A100_PCIe_80GB_aarch64x4):
    num_concurrent_batchers = 6
    num_concurrent_issuers = 6
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GB_aarch64x4_MaxQ(A100_PCIe_80GB_aarch64x4):
    server_target_qps = 92500
    power_limit = 175


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_40GB_ARMx1
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 26000
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1_Triton(A100_PCIe_aarch64x1):
    use_triton = True
    num_concurrent_batchers = 4
    num_concurrent_issuers = 4


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x2(A100_PCIe_aarch64x1):
    system = KnownSystem.A100_PCIe_40GB_ARMx2
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 5
    server_target_qps = 52000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x2_Triton(A100_PCIe_aarch64x2):
    use_triton = True
    num_concurrent_batchers = 4
    num_concurrent_issuers = 4


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x4(A100_PCIe_aarch64x1):
    system = KnownSystem.A100_PCIe_40GB_ARMx4
    gpu_batch_size = 256
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    server_target_qps = 115000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x4_Triton(A100_PCIe_aarch64x4):
    use_triton = True
    num_concurrent_batchers = 6
    num_concurrent_issuers = 6


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_aarch64x4_MaxQ(A100_PCIe_aarch64x4):
    gpu_batch_size = 256
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    server_target_qps = 101000
    power_limit = 175


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_MIG_1x1g_10gb
    use_deque_limit = True
    deque_timeout_usec = 1000
    gpu_batch_size = 8
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    server_target_qps = 3600
    use_graphs = True


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_Hetero(A100_PCIe_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_Triton(A100_PCIe_80GB_MIG_1x1g10gb):
    server_target_qps = 3500
    use_graphs = False
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_56x1g10gb_Triton(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_MIG_56x1g_10gb
    use_deque_limit = True
    deque_timeout_usec = 10
    gpu_batch_size = 4
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    server_target_qps = 70000
    use_graphs = False
    use_triton = True
    instance_group_count = 1
    max_queue_delay_usec = 10
    preferred_batch_size = 1


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_MIG_1x1g_10gb
    use_deque_limit = True
    deque_timeout_usec = 1000
    gpu_batch_size = 8
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    server_target_qps = 3600
    start_from_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Triton(A100_SXM_80GB_MIG_1x1g10gb):
    server_target_qps = 3500
    use_graphs = False
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_56x1g10gb_Triton(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_MIG_56x1g_10gb
    use_deque_limit = True
    deque_timeout_usec = 10
    gpu_batch_size = 4
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    server_target_qps = 70000
    use_graphs = False
    start_from_device = True
    use_triton = True
    instance_group_count = 1
    max_queue_delay_usec = 10
    preferred_batch_size = 1


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx1
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 4
    server_target_qps = 30800
    start_from_device = True
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1_Triton(A100_SXM_80GBx1):
    gpu_copy_streams = 1
    server_target_qps = 30000
    gather_kernel_buffer_threshold = 32
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx4(ServerGPUBaseConfig):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    system = KnownSystem.A100_SXM_80GB_ROx4
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 4
    server_target_qps = 107000
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx4_Triton(A100_SXM_80GBx4):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    deque_timeout_usec = 4000
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    instance_group_count = 2
    batch_triton_requests = False
    server_target_qps = 107000
    gather_kernel_buffer_threshold = 0
    num_concurrent_batchers = 8
    num_concurrent_issuers = 8
    request_timeout_usec = 2000
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx4_MaxQ(A100_SXM_80GBx4):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    power_limit = 225


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx4_Triton_MaxQ(A100_SXM_80GBx4_MaxQ):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    server_target_qps = 80000
    use_graphs = False
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx8
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
class A100_SXM_80GBx8_Triton(A100_SXM_80GBx8):
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


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_MaxQ(A100_SXM_80GBx8):
    server_target_qps = 232000
    power_limit = 225


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_Triton_MaxQ(A100_SXM_80GBx8_MaxQ):
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 200000
    use_graphs = False
    start_from_device = False
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_ARM_MIG_1x1g_10gb
    use_deque_limit = True
    deque_timeout_usec = 1000
    gpu_batch_size = 8
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    server_target_qps = 3600
    use_graphs = True


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Triton(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    server_target_qps = 3500
    use_graphs = False
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x1(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_ARMx1
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 4
    server_target_qps = 30700
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x1_Triton(A100_SXM_80GB_aarch64x1):
    server_target_qps = 22000
    use_triton = True
    gather_kernel_buffer_threshold = 0
    num_concurrent_batchers = 4
    num_concurrent_issuers = 4


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_ARMx8
    use_deque_limit = True
    deque_timeout_usec = 4000
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 260000
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8_Triton(A100_SXM_80GB_aarch64x8):
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    instance_group_count = 2
    batch_triton_requests = False
    server_target_qps = 150000
    gather_kernel_buffer_threshold = 0
    num_concurrent_batchers = 8
    num_concurrent_issuers = 8
    request_timeout_usec = 2000
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM4_40GBx1
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 4
    server_target_qps = 30800
    start_from_device = True
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1_Triton(A100_SXM4_40GBx1):
    gpu_copy_streams = 1
    server_target_qps = 30000
    gather_kernel_buffer_threshold = 32
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx8(A100_SXM4_40GBx1):
    system = KnownSystem.A100_SXM4_40GBx8
    deque_timeout_usec = 4000
    gpu_batch_size = 128
    gpu_inference_streams = 2
    server_target_qps = 255000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx8_Triton(A100_SXM4_40GBx8):
    server_target_qps = 160000
    start_from_device = False
    use_graphs = False
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2(ServerGPUBaseConfig):
    system = KnownSystem.A2x2
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 4
    server_target_qps = 4691
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2_Triton(A2x2):
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
class T4x1(ServerGPUBaseConfig):
    system = KnownSystem.T4x1
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 26
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 4650
    use_cuda_thread_per_device = False
    use_graphs = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1_Triton(T4x1):
    server_target_qps = 4150
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20(T4x1):
    system = KnownSystem.T4x20
    gpu_batch_size = 52
    gpu_copy_streams = 8
    gpu_inference_streams = 4
    server_target_qps = 101000
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20_Triton(T4x20):
    server_target_qps = 97000
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8(T4x1):
    system = KnownSystem.T4x8
    gpu_copy_streams = 8
    server_target_qps = 42000
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8_Triton(T4x8):
    gpu_copy_streams = 8
    server_target_qps = 41200
    use_cuda_thread_per_device = True
    use_triton = True
    use_graphs = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_CPU_2S_6258Rx1_Triton(ServerCPUBaseConfig):
    system = KnownSystem.Triton_CPU_2S_6258R
    use_deque_limit = True
    batch_size = 1
    num_instances = 16
    ov_parameters = {'CPU_THREADS_NUM': '56', 'CPU_THROUGHPUT_STREAMS': '14', 'ENABLE_BATCH_PADDING': 'NO', 'SKIP_OV_DYNAMIC_BATCHSIZE': 'YES'}
    server_target_qps = 2025


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_CPU_4S_8380Hx1_Triton(ServerCPUBaseConfig):
    system = KnownSystem.Triton_CPU_4S_8380H
    use_deque_limit = True
    server_target_qps = 4899.5
    batch_size = 1
    num_instances = 32
    ov_parameters = {'CPU_THREADS_NUM': '112', 'CPU_THROUGHPUT_STREAMS': '28', 'ENABLE_BATCH_PADDING': 'NO', 'SKIP_OV_DYNAMIC_BATCHSIZE': 'YES'}


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_CPU_2S_8380x1_Triton(ServerCPUBaseConfig):
    system = KnownSystem.Triton_CPU_2S_8380
    use_deque_limit = True
    server_target_qps = 3100
    batch_size = 1
    num_instances = 20
    ov_parameters = {'CPU_THREADS_NUM': '80', 'CPU_THROUGHPUT_STREAMS': '20', 'ENABLE_BATCH_PADDING': 'NO', 'SKIP_OV_DYNAMIC_BATCHSIZE': 'YES'}
