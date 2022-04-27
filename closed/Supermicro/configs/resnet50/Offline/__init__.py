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
from configs.resnet50 import GPUBaseConfig, CPUBaseConfig


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline

    run_infer_on_copy_streams = False
    use_graphs = False
    gpu_inference_streams = 1
    gpu_copy_streams = 2


class OfflineCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Offline


# AS-4124GO-NART_8_A100-SXM-80GB_TRT
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx8
    gpu_batch_size = 2048
    start_from_device = True

    run_infer_on_copy_streams = False
    gpu_inference_streams = 2
    gpu_copy_streams = 4
    offline_expected_qps = 323400

    numa_config = "0,1,2,3:0-31,64-95&4,5,6,7:32-63,96-127"


# SYS-220HE-FTNR_4_A2_TRT 
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x4(OfflineGPUBaseConfig):
    system = KnownSystem.A2x4
    gpu_batch_size = 1024
    offline_expected_qps = 12088
    run_infer_on_copy_streams = None


# SYS-220HE-FTNR_2_A2_TRT
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2(OfflineGPUBaseConfig):
    system = KnownSystem.A2x2
    gpu_batch_size = 1024
    offline_expected_qps = 6044
    run_infer_on_copy_streams = None


# SYS-220HE-FTNR_1_A2_TRT
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1(OfflineGPUBaseConfig):
    system = KnownSystem.A2x1
    gpu_batch_size = 1024
    offline_expected_qps = 3022
    run_infer_on_copy_streams = None


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS_2124GQ_NART(OfflineGPUBaseConfig):
    _system_alias = "AS_2124GQ_NART_Redstone"
    _notes = "This is AS_2124GQ_NART"

    system = KnownSystem.AS_2124GQ_NART
    gpu_batch_size = 2048
    offline_expected_qps = 280000
    numa_config = "1:0-31,128-159&0:32-63,160-191&3:64-95,192-223&2:96-127,224-255"


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS_2124GQ_NART_Triton(AS_2124GQ_NART):
    _system_alias = "AS_2124GQ_NART_Redstone"
    _notes = "This is AS_2124GQ_NART"

    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class AS_2124GQ_NART_MaxQ(AS_2124GQ_NART):
    _system_alias = "AS_2124GQ_NART_Redstone"
    _notes = "This is AS_2124GQ_NART"

    power_limit = 225


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class AS_2124GQ_NART_Triton_MaxQ(AS_2124GQ_NART_Triton):
    _system_alias = "AS_2124GQ_NART_Redstone"
    _notes = "This is AS_2124GQ_NART"

    power_limit = 225




