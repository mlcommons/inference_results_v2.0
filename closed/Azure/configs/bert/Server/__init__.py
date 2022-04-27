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
from configs.bert import GPUBaseConfig, CPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server

    enable_interleaved = False
    use_graphs = True
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120


class ServerCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Server


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC48ads_A100_v4(ServerGPUBaseConfig):
    system = KnownSystem.NC48ads_A100_v4
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 5500 
    soft_drop = 1.0


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC48ads_A100_v4_HighAccuracy(NC48ads_A100_v4):
    precision = "fp16"
    server_target_qps = 2750


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC48ads_A100_v4_Triton(NC48ads_A100_v4):
    server_target_qps = 5200 
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC48ads_A100_v4_HighAccuracy_Triton(NC48ads_A100_v4_HighAccuracy):
    server_target_qps = 2700
    use_triton = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4(ServerGPUBaseConfig):
    system = KnownSystem.NC96ads_A100_v4
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 11500
    soft_drop = 1.0


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC96ads_A100_v4_HighAccuracy(NC96ads_A100_v4):
    precision = "fp16"
    server_target_qps = 5400


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4_Triton(NC96ads_A100_v4):
    server_target_qps = 10350
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC96ads_A100_v4_HighAccuracy_Triton(NC96ads_A100_v4_HighAccuracy):
    server_target_qps = 5263
    use_triton = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(ServerGPUBaseConfig):
    system = KnownSystem.A30_MIG_1x1g_6gb
    gpu_inference_streams = 1
    active_sms = 60
    gpu_batch_size = 10
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 380
    soft_drop = 0.99
    deque_timeout_usec = 50000
    workspace_size = 805306368
    use_graphs = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy(A30_MIG_1x1g6gb):
    precision = "fp16"
    gpu_batch_size = 6
    server_target_qps = 150


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    server_target_qps = 340


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero_HighAccuracy(A30_MIG_1x1g6gb_HighAccuracy):
    server_target_qps = 130


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(A30_MIG_1x1g6gb):
    server_target_qps = 260
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy_Triton(A30_MIG_1x1g6gb_HighAccuracy):
    gpu_batch_size = 64
    server_target_qps = 120
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_32x1g6gb_Triton(A30_MIG_1x1g6gb):
    system = KnownSystem.A30_MIG_32x1g_6gb
    deque_timeout_usec = 1000
    server_target_qps = 8300
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_32x1g6gb_HighAccuracy_Triton(A30_MIG_32x1g6gb_Triton):
    precision = "fp16"
    gpu_batch_size = 6
    server_target_qps = 3800


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(ServerGPUBaseConfig):
    system = KnownSystem.A30x1
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 1500
    soft_drop = 0.993


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x1_HighAccuracy(A30x1):
    precision = "fp16"
    server_target_qps = 650


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_Triton(A30x1):
    server_target_qps = 1400
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x1_HighAccuracy_Triton(A30x1_HighAccuracy):
    precision = "fp16"
    server_target_qps = 600
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8(A30x1):
    system = KnownSystem.A30x8
    server_target_qps = 11500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x8_HighAccuracy(A30x8):
    precision = "fp16"
    server_target_qps = 5250


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8_Triton(A30x8):
    server_target_qps = 11000
    use_triton = True
    max_queue_delay_usec = 1000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x8_HighAccuracy_Triton(A30x8_HighAccuracy):
    server_target_qps = 5200
    use_triton = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96amsr_A100_v4(ServerGPUBaseConfig):
    system = KnownSystem.ND96amsr_A100_v4
    active_sms = 60
    gpu_batch_size = 48
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 25800
    soft_drop = 0.99


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ND96amsr_A100_v4_HighAccuracy(ND96amsr_A100_v4):
    gpu_batch_size = 24
    precision = "fp16"
    server_target_qps = 13100


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96amsr_A100_v4_Triton(ND96amsr_A100_v4):
    server_target_qps = 22400
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ND96amsr_A100_v4_HighAccuracy_Triton(ND96amsr_A100_v4_HighAccuracy):
    gpu_batch_size = 64
    server_target_qps = 11205
    use_triton = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96asr_v4(ServerGPUBaseConfig):
    system = KnownSystem.ND96asr_v4
    active_sms = 60
    gpu_batch_size = 96
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 25000
    soft_drop = 0.99


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ND96asr_v4_HighAccuracy(ND96asr_v4):
    gpu_batch_size = 64
    precision = "fp16"
    server_target_qps = 11500


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96asr_v4_Triton(ND96asr_v4):
    server_target_qps = 22455
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ND96asr_v4_HighAccuracy_Triton(ND96asr_v4_HighAccuracy):
    server_target_qps = 11205
    use_triton = True
