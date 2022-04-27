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


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    use_graphs = True
    gpu_inference_streams = 1
    gpu_copy_streams = 1
    num_warmups = 20480
    nobatch_sorting = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC48ads_A100_v4(ServerGPUBaseConfig):
    system = KnownSystem.NC48ads_A100_v4
    gpu_batch_size = 2048
    server_target_qps = 23000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4(ServerGPUBaseConfig):
    system = KnownSystem.NC96ads_A100_v4
    gpu_batch_size = 2048
    server_target_qps = 46400

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(ServerGPUBaseConfig):
    system = KnownSystem.A30_MIG_1x1g_6gb
    audio_batch_size = 32
    audio_buffer_num_lines = 512
    audio_fp16_input = None
    dali_batches_issue_ahead = 1
    dali_pipeline_depth = 1
    gpu_batch_size = 256
    num_warmups = 32
    nobatch_sorting = None
    server_target_qps = 1100
    workspace_size = 1610612736


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    server_target_qps = 950


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(ServerGPUBaseConfig):
    system = KnownSystem.A30x1
    gpu_batch_size = 1792
    server_target_qps = 5200


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8(ServerGPUBaseConfig):
    system = KnownSystem.A30x8
    gpu_batch_size = 1792
    server_target_qps = 37000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A30x8_MaxQ(A30x8):
    server_target_qps = 43500.0
    power_limit = 200

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96amsr_A100_v4(ServerGPUBaseConfig):
    system = KnownSystem.ND96amsr_A100_v4
    start_from_device = True
    dali_pipeline_depth = 1
    gpu_batch_size = 2048
    server_num_issue_query_threads = 0
    server_target_qps = 104000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96asr_v4(ServerGPUBaseConfig):
    system = KnownSystem.ND96asr_v4
    dali_pipeline_depth = 1
    gpu_batch_size = 2048
    server_num_issue_query_threads = 0
    server_target_qps = 102050 
    start_from_device = True
