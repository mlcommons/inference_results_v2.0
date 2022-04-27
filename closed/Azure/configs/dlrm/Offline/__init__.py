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


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    check_contiguity = True
    use_small_tile_gemm_plugin = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC24ads_A100_v4(OfflineGPUBaseConfig):
    system = KnownSystem.NC24ads_A100_v4
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 315000
    offline_expected_qps = 270000
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC24ads_A100_v4_HighAccuracy(NC24ads_A100_v4):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC24ads_A100_v4_Triton(NC24ads_A100_v4):
    batch_triton_requests = True
    buffer_manager_thread_count = 8
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC24ads_A100_v4_HighAccuracy_Triton(NC24ads_A100_v4_Triton):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC48ads_A100_v4(OfflineGPUBaseConfig):
    system = KnownSystem.NC48ads_A100_v4
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 315000
    offline_expected_qps = 270000 * 2
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC48ads_A100_v4_HighAccuracy(NC48ads_A100_v4):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC48ads_A100_v4_Triton(NC48ads_A100_v4):
    batch_triton_requests = True
    buffer_manager_thread_count = 8
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC48ads_A100_v4_HighAccuracy_Triton(NC48ads_A100_v4_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4(OfflineGPUBaseConfig):
    system = KnownSystem.NC96ads_A100_v4
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 315000
    offline_expected_qps = 270000 * 4
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC96ads_A100_v4_HighAccuracy(NC96ads_A100_v4):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4_Triton(NC96ads_A100_v4):
    batch_triton_requests = True
    buffer_manager_thread_count = 8
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC96ads_A100_v4_HighAccuracy_Triton(NC96ads_A100_v4_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(OfflineGPUBaseConfig):
    system = KnownSystem.A30_MIG_1x1g_6gb
    use_small_tile_gemm_plugin = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 12800
    offline_expected_qps = 34000
    max_pairs_per_staging_thread = 12800
    num_staging_batches = 2
    num_staging_threads = 2
    use_jemalloc = True
    workspace_size = 536870912


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy(A30_MIG_1x1g6gb):
    pass


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    offline_expected_qps = 31117


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero_HighAccuracy(A30_MIG_1x1g6gb_Hetero):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(A30_MIG_1x1g6gb):
    batch_triton_requests = True
    buffer_manager_thread_count = 8
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_32x1g6gb_Triton(A30_MIG_1x1g6gb):
    system = KnownSystem.A30_MIG_32x1g_6gb
    offline_expected_qps = 1088000
    batch_triton_requests = True
    buffer_manager_thread_count = 8
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(OfflineGPUBaseConfig):
    system = KnownSystem.A30x1
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    offline_expected_qps = 140000
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x1_HighAccuracy(A30x1):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_Triton(A30x1):
    batch_triton_requests = True
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x1_HighAccuracy_Triton(A30x1_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8(A30x1):
    system = KnownSystem.A30x8
    offline_expected_qps = 1120000
    numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x8_HighAccuracy(A30x8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8_Triton(A30x8):
    batch_triton_requests = True
    use_triton = True
    num_concurrent_batchers = 1


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x8_HighAccuracy_Triton(A30x8_Triton):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96amsr_A100_v4(OfflineGPUBaseConfig):
    system = KnownSystem.ND96amsr_A100_v4
    complete_threads = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 334000
    offline_expected_qps = 2400000
    max_pairs_per_staging_thread = 262100
    start_from_device = True
    use_jemalloc = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ND96amsr_A100_v4_HighAccuracy(ND96amsr_A100_v4):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96amsr_A100_v4_Triton(ND96amsr_A100_v4):
    gpu_batch_size = 262100
    offline_expected_qps = 2450000
    batch_triton_requests = True
    use_triton = True
    num_concurrent_batchers = 1


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ND96amsr_A100_v4_HighAccuracy_Triton(ND96amsr_A100_v4_Triton):
    gpu_batch_size = 334000
    offline_expected_qps = 2400000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96asr_v4(OfflineGPUBaseConfig):
    system = KnownSystem.ND96asr_v4
    complete_threads = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    offline_expected_qps = 2120000
    max_pairs_per_staging_thread = 262100
    start_from_device = True
    use_jemalloc = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ND96asr_v4_HighAccuracy(ND96asr_v4):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96asr_v4_Triton(ND96asr_v4):
    offline_expected_qps = 1900000
    batch_triton_requests = True
    use_triton = True
    num_concurrent_batchers = 1


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ND96asr_v4_HighAccuracy_Triton(ND96asr_v4_Triton):
    pass
