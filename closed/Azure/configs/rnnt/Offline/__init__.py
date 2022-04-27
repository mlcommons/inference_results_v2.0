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
from configs.rnnt import GPUBaseConfig


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    use_graphs = True
    num_warmups = 512
    audio_batch_size = 512
    audio_buffer_num_lines = 4096
    dali_batches_issue_ahead = 4
    dali_pipeline_depth = 4
    gpu_inference_streams = 1
    gpu_copy_streams = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC24ads_A100_v4(OfflineGPUBaseConfig):
    system = KnownSystem.NC24ads_A100_v4
    gpu_batch_size = 2048
    offline_expected_qps = 13300


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC48ads_A100_v4(OfflineGPUBaseConfig):
    system = KnownSystem.NC48ads_A100_v4
    gpu_batch_size = 2048
    offline_expected_qps = 13300 * 2

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4(OfflineGPUBaseConfig):
    system = KnownSystem.NC96ads_A100_v4
    gpu_batch_size = 2048
    offline_expected_qps = 13300 * 4

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(OfflineGPUBaseConfig):
    system = KnownSystem.A30_MIG_1x1g_6gb
    audio_batch_size = 32
    audio_buffer_num_lines = 512
    dali_batches_issue_ahead = 1
    dali_pipeline_depth = 1
    gpu_batch_size = 1024
    offline_expected_qps = 1450
    max_seq_length = 64
    num_warmups = 32
    workspace_size = 1610612736


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    offline_expected_qps = 1337


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(OfflineGPUBaseConfig):
    system = KnownSystem.A30x1
    gpu_batch_size = 2048
    offline_expected_qps = 6959.999999999999


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8(A30x1):
    system = KnownSystem.A30x8
    offline_expected_qps = 55679.99999999999


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A30x8_MaxQ(A30x8):
    offline_expected_qps = 46400.0
    power_limit = 250

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96amsr_A100_v4(OfflineGPUBaseConfig):
    system = KnownSystem.ND96amsr_A100_v4
    gpu_batch_size = 2048
    offline_expected_qps = 104000
    start_from_device = True
    audio_batch_size = 1024
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    num_warmups = 40480
    nobatch_sorting = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96asr_v4(OfflineGPUBaseConfig):
    system = KnownSystem.ND96asr_v4
    gpu_batch_size = 2048
    offline_expected_qps = 100000
    start_from_device = True
    num_warmups = 40480
    audio_buffer_num_lines = None
