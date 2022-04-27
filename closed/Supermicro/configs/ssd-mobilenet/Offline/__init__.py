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


GPUBaseConfig = import_module("configs.ssd-mobilenet").GPUBaseConfig


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    use_graphs = False


# SYS-220HE-FTNR_2_A2_TRT
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2(OfflineGPUBaseConfig):
    system = KnownSystem.A2x2
    gpu_inference_streams = 1
    gpu_batch_size = 768
    gpu_copy_streams = 2
    input_format = "chw4"
    tensor_path = "build/preprocessed_data/coco/val2017/SSDMobileNet/int8_chw4"
    offline_expected_qps = 9200


# SYS-220HE-FTNR_1_A2_TRT
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1(OfflineGPUBaseConfig):
    system = KnownSystem.A2x1
    gpu_inference_streams = 1
    gpu_batch_size = 768
    gpu_copy_streams = 2
    input_format = "chw4"
    tensor_path = "build/preprocessed_data/coco/val2017/SSDMobileNet/int8_chw4"
    offline_expected_qps = 4600


