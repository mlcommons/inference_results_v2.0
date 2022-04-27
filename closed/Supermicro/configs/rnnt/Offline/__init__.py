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


# AS-4124GO-NART_8_A100-SXM-80GB_TRT
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx8
    gpu_batch_size = 2048
    start_from_device = True

    audio_batch_size = 1024
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    gpu_copy_streams = 4
    offline_expected_qps = 112000
    num_warmups = 40480
    nobatch_sorting = True


# SYS-220HE-FTNR_4_A2_TRT
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x4(OfflineGPUBaseConfig):
    system = KnownSystem.A2x4
    dali_batches_issue_ahead = 2
    dali_pipeline_depth = 2
    gpu_batch_size = 512
    audio_batch_size = 128
    offline_expected_qps = 4080


# SYS-220HE-FTNR_2_A2_TRT
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2(OfflineGPUBaseConfig):
    system = KnownSystem.A2x2
    dali_batches_issue_ahead = 2
    dali_pipeline_depth = 2
    gpu_batch_size = 512
    audio_batch_size = 128
    offline_expected_qps = 2000


# SYS-220HE-FTNR_1_A2_TRT
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1(OfflineGPUBaseConfig):
    system = KnownSystem.A2x1
    dali_batches_issue_ahead = 2
    dali_pipeline_depth = 2
    gpu_batch_size = 512
    audio_batch_size = 128
    offline_expected_qps = 1000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS_2124GQ_NART(OfflineGPUBaseConfig):
    _system_alias = "AS_2124GQ_NART_Redstone"
    _notes = "This is AS_2124GQ_NART"

    system = KnownSystem.AS_2124GQ_NART
    gpu_batch_size = 2048
    offline_expected_qps = 888000
    numa_config = "1:0-31,128-159&0:32-63,160-191&3:64-95,192-223&2:96-127,224-255"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class AS_2124GQ_NART_MaxQ(AS_2124GQ_NART):
    _system_alias = "AS_2124GQ_NART_Redstone"
    _notes = "This is AS_2124GQ_NART"

    power_limit = 225


