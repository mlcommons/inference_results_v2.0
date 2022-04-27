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


# AS-4124GO-NART_8_A100-SXM-80GB_TRT
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx8
    active_sms = 60
    gpu_batch_size = 92
    gpu_copy_streams = 4
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 25600
    soft_drop = 0.99


# AS-4124GO-NART_8_A100-SXM-80GB_TRT
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx8_HighAccuracy(A100_SXM_80GBx8):
    gpu_batch_size = 24
    precision = "fp16"
    server_target_qps = 127200


# SYS-220HE-FTNR_4_A2_TRT
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x4(ServerGPUBaseConfig):
    system = KnownSystem.A2x4
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    #active_sms = 10
    gpu_batch_size = 4
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    soft_drop = 0.993

    server_target_qps = 685


# SYS-220HE-FTNR_4_A2_TRT
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A2x4_HighAccuracy(A2x4):
    precision = "fp16"
    gpu_batch_size = 8
    server_target_qps = 282


