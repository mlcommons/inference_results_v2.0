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


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline

    gpu_copy_streams = 2
    gpu_inference_streams = 2
    enable_interleaved = False


class OfflineCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Offline

    max_queue_delay_usec = 100


# AS-4124GO-NART_8_A100-SXM-80GB_TRT
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx8
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1280
    gpu_inference_streams = 1

    gpu_copy_streams = 4
    offline_expected_qps = 30800
    workspace_size = 7516192768


# AS-4124GO-NART_8_A100-SXM-80GB_TRT
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx8_HighAccuracy(A100_SXM_80GBx8):
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 15900


# SYS-220HE-FTNR_4_A2_TRT
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x4(OfflineGPUBaseConfig):
    system = KnownSystem.A2x4
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 256
    offline_expected_qps = 1000


# SYS-220HE-FTNR_4_A2_TRT
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A2x4_HighAccuracy(A2x4):
    precision = "fp16"
    gpu_inference_streams = 1
    offline_expected_qps = 480


# SYS-220HE-FTNR_2_A2_TRT
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2(OfflineGPUBaseConfig):
    system = KnownSystem.A2x2
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 256
    offline_expected_qps = 500


# SYS-220HE-FTNR_1_A2_TRT
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1(OfflineGPUBaseConfig):
    system = KnownSystem.A2x1
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 256
    offline_expected_qps = 250



