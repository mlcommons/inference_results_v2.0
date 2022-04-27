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
from configs.resnet50 import GPUBaseConfig


class MultiStreamGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.MultiStream
    gpu_batch_size = 8
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    multi_stream_samples_per_query = 8
    multi_stream_target_latency_percentile = 99
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC24ads_A100_v4(MultiStreamGPUBaseConfig):
    system = KnownSystem.NC24ads_A100_v4
    multi_stream_expected_latency_ns = 830000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC24ads_A100_v4_Triton(NC24ads_A100_v4):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(MultiStreamGPUBaseConfig):
    system = KnownSystem.A30_MIG_1x1g_6gb
    multi_stream_expected_latency_ns = 5726152


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    multi_stream_expected_latency_ns = 5968320


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(A30_MIG_1x1g6gb):
    use_graphs = False
    use_triton = True
    multi_stream_expected_latency_ns = 6889088


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_32x1g6gb_Triton(MultiStreamGPUBaseConfig):
    system = KnownSystem.A30_MIG_32x1g_6gb
    use_graphs = False
    use_triton = True
    multi_stream_expected_latency_ns = 6889088


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(MultiStreamGPUBaseConfig):
    system = KnownSystem.A30x1
    multi_stream_expected_latency_ns = 960000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_Triton(A30x1):
    use_graphs = False
    use_triton = True
    multi_stream_expected_latency_ns = 1330000

