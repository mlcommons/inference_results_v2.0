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

ParentConfig = import_module("configs.3d-unet")
GPUBaseConfig = ParentConfig.GPUBaseConfig
CPUBaseConfig = ParentConfig.CPUBaseConfig


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    gpu_inference_streams = 1
    gpu_copy_streams = 1


class OfflineCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Offline


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC24ads_A100_v4(OfflineGPUBaseConfig):
    system = KnownSystem.NC24ads_A100_v4
    gpu_batch_size = 1
    offline_expected_qps = 2.6


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC24ads_A100_v4_HighAccuracy(NC24ads_A100_v4):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC24ads_A100_v4_Triton(NC24ads_A100_v4):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC24ads_A100_v4_HighAccuracy_Triton(NC24ads_A100_v4_Triton):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC48ads_A100_v4(OfflineGPUBaseConfig):
    system = KnownSystem.NC48ads_A100_v4
    gpu_batch_size = 1
    offline_expected_qps = 2.6 * 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC48ads_A100_v4_HighAccuracy(NC48ads_A100_v4):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC48ads_A100_v4_Triton(NC48ads_A100_v4):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC48ads_A100_v4_HighAccuracy_Triton(NC48ads_A100_v4_Triton):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4(OfflineGPUBaseConfig):
    system = KnownSystem.NC96ads_A100_v4
    gpu_batch_size = 1
    offline_expected_qps = 2.6 * 4


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC96ads_A100_v4_HighAccuracy(NC96ads_A100_v4):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4_Triton(NC96ads_A100_v4):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC96ads_A100_v4_HighAccuracy_Triton(NC96ads_A100_v4_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(OfflineGPUBaseConfig):
    system = KnownSystem.A30_MIG_1x1g_6gb
    workspace_size = 805306368
    gpu_batch_size = 1
    offline_expected_qps = 0.04


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy(A30_MIG_1x1g6gb):
    pass


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    offline_expected_qps = 0.3


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero_HighAccuracy(A30_MIG_1x1g6gb_Hetero):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(A30_MIG_1x1g6gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy_Triton(A30_MIG_1x1g6gb_Triton):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_32x1g6gb_Triton(OfflineGPUBaseConfig):
    system = KnownSystem.A30_MIG_32x1g_6gb
    workspace_size = 805306368
    gpu_batch_size = 1
    offline_expected_qps = 1.2
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_32x1g6gb_HighAccuracy_Triton(A30_MIG_32x1g6gb_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(OfflineGPUBaseConfig):
    system = KnownSystem.A30x1
    gpu_batch_size = 1
    offline_expected_qps = 1.64


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x1_HighAccuracy(A30x1):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_Triton(A30x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x1_HighAccuracy_Triton(A30x1_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8(A30x1):
    system = KnownSystem.A30x8
    offline_expected_qps = 13
    numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x8_HighAccuracy(A30x8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8_Triton(A30x8):
    offline_expected_qps = 13
    use_triton = True
    output_pinned_memory = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x8_HighAccuracy_Triton(A30x8_Triton):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96amsr_A100_v4(OfflineGPUBaseConfig):
    system = KnownSystem.ND96amsr_A100_v4
    gpu_batch_size = 1
    offline_expected_qps = 24
    start_from_device = True
    end_on_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ND96amsr_A100_v4_HighAccuracy(ND96amsr_A100_v4):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96amsr_A100_v4_Triton(ND96amsr_A100_v4):
    use_graphs = True
    instance_group_count = 4
    use_triton = True
    output_pinned_memory = False

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ND96amsr_A100_v4_HighAccuracy_Triton(ND96amsr_A100_v4_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96asr_v4(OfflineGPUBaseConfig):
    system = KnownSystem.ND96asr_v4
    gpu_batch_size = 1
    start_from_device = True
    offline_expected_qps = 23


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ND96asr_v4_HighAccuracy(ND96asr_v4):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND96asr_v4_Triton(ND96asr_v4):
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ND96asr_v4_HighAccuracy_Triton(ND96asr_v4_Triton):
    pass
