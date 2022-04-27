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


class SingleStreamGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.SingleStream

    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    use_graphs = True
    bert_opt_seqlen = 270
    use_small_tile_gemm_plugin = False

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC24ads_A100_v4(SingleStreamGPUBaseConfig):
    system = KnownSystem.NC24ads_A100_v4
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 1700000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC24ads_A100_v4_HighAccuracy(NC24ads_A100_v4):
    precision = "fp16"
    single_stream_expected_latency_ns = 1700000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC24ads_A100_v4_Triton(NC24ads_A100_v4):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC24ads_A100_v4_HighAccuracy_Triton(NC24ads_A100_v4_HighAccuracy):
    use_triton = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(SingleStreamGPUBaseConfig):
    system = KnownSystem.A30_MIG_1x1g_6gb
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 5999404
    workspace_size = 1610612736


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy(A30_MIG_1x1g6gb):
    precision = "fp16"
    single_stream_expected_latency_ns = 11000950


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    single_stream_expected_latency_ns = 6412655


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero_HighAccuracy(A30_MIG_1x1g6gb_HighAccuracy):
    single_stream_expected_latency_ns = 11725740


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(A30_MIG_1x1g6gb):
    single_stream_expected_latency_ns = 5999404
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy_Triton(A30_MIG_1x1g6gb_Triton):
    precision = "fp16"
    single_stream_expected_latency_ns = 7452826


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_32x1g6gb_Triton(SingleStreamGPUBaseConfig):
    system = KnownSystem.A30_MIG_32x1g_6gb
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    workspace_size = 1610612736
    single_stream_expected_latency_ns = 9400000
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_32x1g6gb_HighAccuracy_Triton(A30_MIG_32x1g6gb_Triton):
    precision = "fp16"
    single_stream_expected_latency_ns = 7452826


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.A30x1
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 1700000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x1_HighAccuracy(A30x1):
    precision = "fp16"


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_Triton(A30x1):
    single_stream_expected_latency_ns = 3400000
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x1_HighAccuracy_Triton(A30x1_Triton):
    precision = "fp16"
