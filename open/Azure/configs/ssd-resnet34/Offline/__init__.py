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


ParentConfig = import_module("configs.ssd-resnet34")
GPUBaseConfig = ParentConfig.GPUBaseConfig
CPUBaseConfig = ParentConfig.CPUBaseConfig


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    use_graphs = False


class OfflineCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Offline
    batch_size = 1
    run_infer_on_copy_streams = False


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC24ads_A100_v4(OfflineGPUBaseConfig):
    system = KnownSystem.NC24ads_A100_v4
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    offline_expected_qps = 960
    run_infer_on_copy_streams = False

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC24ads_A100_v4_Triton(NC24ads_A100_v4):
    use_triton = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC48ads_A100_v4(OfflineGPUBaseConfig):
    system = KnownSystem.NC48ads_A100_v4
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    offline_expected_qps = 960 * 2
    run_infer_on_copy_streams = False

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC48ads_A100_v4_Triton(NC48ads_A100_v4):
    use_triton = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4(OfflineGPUBaseConfig):
    system = KnownSystem.NC96ads_A100_v4
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    offline_expected_qps = 960 * 4
    run_infer_on_copy_streams = False

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4_Triton(NC96ads_A100_v4):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(OfflineGPUBaseConfig):
    system = KnownSystem.A30_MIG_1x1g_6gb
    gpu_batch_size = 4
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    workspace_size = 1610612736
    offline_expected_qps = 128


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    offline_expected_qps = 115


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(A30_MIG_1x1g6gb):
    use_triton = True
    offline_expected_qps = 130


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_32x1g6gb_Triton(OfflineGPUBaseConfig):
    system = KnownSystem.A30_MIG_32x1g_6gb
    gpu_batch_size = 4
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    workspace_size = 1610612736
    use_triton = True
    offline_expected_qps = 4160


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(OfflineGPUBaseConfig):
    system = KnownSystem.A30x1
    gpu_batch_size = 32
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    offline_expected_qps = 470
    run_infer_on_copy_streams = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_Triton(A30x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8(A30x1):
    system = KnownSystem.A30x8
    offline_expected_qps = 3760.0


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8_Triton(A30x8):
    use_triton = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96amsr_A100_v4(OfflineGPUBaseConfig):
    system = KnownSystem.ND96amsr_A100_v4
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    start_from_device = True
    offline_expected_qps = 7800
    run_infer_on_copy_streams = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96amsr_A100_v4_Triton(ND96amsr_A100_v4):
    start_from_device = None
    use_triton = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96asr_v4(OfflineGPUBaseConfig):
    system = KnownSystem.ND96asr_v4
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 1
    start_from_device = True
    offline_expected_qps = 7600


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96asr_v4_Triton(ND96asr_v4):
    start_from_device = None
    use_triton = True

