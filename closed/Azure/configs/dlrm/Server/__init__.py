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

ParentConfig = import_module("configs.dlrm")
GPUBaseConfig = ParentConfig.GPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    complete_threads = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC24ads_A100_v4(ServerGPUBaseConfig):
    system = KnownSystem.NC24ads_A100_v4
    deque_timeout_usec = 1
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 180000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC24ads_A100_v4_HighAccuracy(NC24ads_A100_v4):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC24ads_A100_v4_Triton(NC24ads_A100_v4):
    batch_triton_requests = True
    buffer_manager_thread_count = 8
    max_queue_delay_usec = 10000
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC24ads_A100_v4_HighAccuracy_Triton(NC24ads_A100_v4_Triton):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC48ads_A100_v4(ServerGPUBaseConfig):
    system = KnownSystem.NC48ads_A100_v4
    deque_timeout_usec = 1
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 180000 * 2
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC48ads_A100_v4_HighAccuracy(NC48ads_A100_v4):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC48ads_A100_v4_Triton(NC48ads_A100_v4):
    batch_triton_requests = True
    buffer_manager_thread_count = 8
    max_queue_delay_usec = 10000
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC48ads_A100_v4_HighAccuracy_Triton(NC48ads_A100_v4_Triton):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4(ServerGPUBaseConfig):
    system = KnownSystem.NC96ads_A100_v4
    deque_timeout_usec = 1
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 180000 * 4
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC96ads_A100_v4_HighAccuracy(NC96ads_A100_v4):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4_Triton(NC96ads_A100_v4):
    batch_triton_requests = True
    buffer_manager_thread_count = 8
    max_queue_delay_usec = 10000
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC96ads_A100_v4_HighAccuracy_Triton(NC96ads_A100_v4_Triton):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(ServerGPUBaseConfig):
    system = KnownSystem.A30_MIG_1x1g_6gb
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.02
    gpu_batch_size = 226000
    gpu_num_bundles = 2
    num_staging_batches = 2
    num_staging_threads = 4
    server_target_qps = 31000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy(A30_MIG_1x1g6gb):
    pass


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    server_target_qps = 30000


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero_HighAccuracy(A30_MIG_1x1g6gb_Hetero):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(A30_MIG_1x1g6gb):
    server_target_qps = 25000.0
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy_Triton(A30_MIG_1x1g6gb_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(ServerGPUBaseConfig):
    system = KnownSystem.A30x1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gpu_batch_size = 226000
    gpu_num_bundles = 2
    num_staging_batches = 2
    num_staging_threads = 4
    server_target_qps = 132000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x1_HighAccuracy(A30x1):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_Triton(A30x1):
    server_target_qps = 100000
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x1_HighAccuracy_Triton(A30x1_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8(A30x1):
    system = KnownSystem.A30x8
    gpu_batch_size = 131000
    num_staging_batches = 8
    num_staging_threads = 8
    server_target_qps = 1000000
    use_jemalloc = False
    numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x8_HighAccuracy(A30x8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8_Triton(A30x8):
    server_target_qps = 600000
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x8_HighAccuracy_Triton(A30x8_Triton):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96amsr_A100_v4(ServerGPUBaseConfig):
    system = KnownSystem.ND96amsr_A100_v4
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 2300000
    start_from_device = True
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ND96amsr_A100_v4_HighAccuracy(ND96amsr_A100_v4):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96amsr_A100_v4_Triton(ServerGPUBaseConfig):
    system = KnownSystem.ND96amsr_A100_v4
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 725000
    use_jemalloc = True
    max_queue_delay_usec = 1000
    use_triton = True
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64
    request_timeout_usec = 2000
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ND96amsr_A100_v4_HighAccuracy_Triton(ND96amsr_A100_v4_Triton):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96asr_v4(ServerGPUBaseConfig):
    system = KnownSystem.ND96asr_v4
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 2100000
    start_from_device = True
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ND96asr_v4_HighAccuracy(ND96asr_v4):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96asr_v4_Triton(ServerGPUBaseConfig):
    system = KnownSystem.ND96asr_v4
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 80000
    use_jemalloc = True
    max_queue_delay_usec = 10000
    use_triton = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ND96asr_v4_HighAccuracy_Triton(ND96asr_v4_Triton):
    pass
