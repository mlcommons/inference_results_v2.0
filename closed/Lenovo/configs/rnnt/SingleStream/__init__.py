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


class SingleStreamGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.SingleStream
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    audio_batch_size = 1
    audio_fp16_input = True
    dali_batches_issue_ahead = 1
    dali_pipeline_depth = 1
    use_graphs = True
    disable_encoder_plugin = True
    nobatch_sorting = True
    num_warmups = 32


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex1(SingleStreamGPUBaseConfig):
    system = KnownSystem.A100_PCIe_40GBx1
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1(SingleStreamGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBx1
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_ARMx1
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.A100_PCIe_40GB_ARMx1
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb(SingleStreamGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_MIG_1x1g_10gb
    audio_buffer_num_lines = 4
    single_stream_expected_latency_ns = 78000000
    nouse_copy_kernel = True
    workspace_size = 1073741824


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_Hetero(A100_PCIe_80GB_MIG_1x1g10gb):
    single_stream_expected_latency_ns = 80000000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb(SingleStreamGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_MIG_1x1g_10gb
    audio_buffer_num_lines = 4
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = True
    workspace_size = 1073741824


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1(SingleStreamGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx1
    audio_buffer_num_lines = 4
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = True
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb(SingleStreamGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_ARM_MIG_1x1g_10gb
    audio_buffer_num_lines = 4
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = True
    workspace_size = 1073741824


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_ARMx1
    audio_buffer_num_lines = 4
    single_stream_expected_latency_ns = 38400000
    nouse_copy_kernel = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb(SingleStreamGPUBaseConfig):
    system = KnownSystem.A100_SXM4_40GB_MIG_1x1g_5gb
    audio_buffer_num_lines = 4
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = True
    workspace_size = 1073741824


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1(SingleStreamGPUBaseConfig):
    system = KnownSystem.A100_SXM4_40GBx1
    audio_buffer_num_lines = 4
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = True
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.A2x1
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 105000000
    nouse_copy_kernel = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(SingleStreamGPUBaseConfig):
    system = KnownSystem.A30_MIG_1x1g_6gb
    audio_batch_size = 32
    audio_buffer_num_lines = 512
    single_stream_expected_latency_ns = 76133687
    nouse_copy_kernel = False
    workspace_size = 1610612736


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    single_stream_expected_latency_ns = 78812921


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.A30x1
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = False

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x2(A30x1):
    system = KnownSystem.A30x2
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 5000000
    nouse_copy_kernel = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class AGX_Xavier(SingleStreamGPUBaseConfig):
    system = KnownSystem.AGX_Xavier
    single_stream_expected_latency_ns = 100000000
    audio_fp16_input = None
    dali_batches_issue_ahead = None
    dali_pipeline_depth = None
    nobatch_sorting = None
    num_warmups = None


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class AGX_Xavier_MaxQ(AGX_Xavier):
    # power settings
    soc_gpu_freq = 1032750000
    soc_dla_freq = 115200000
    soc_cpu_freq = 1190400
    soc_emc_freq = 1600000000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.T4x1
    audio_buffer_num_lines = 4
    single_stream_expected_latency_ns = 25000000
    nouse_copy_kernel = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class Xavier_NX(SingleStreamGPUBaseConfig):
    system = KnownSystem.Xavier_NX
    single_stream_expected_latency_ns = 200000000
    audio_fp16_input = None
    dali_batches_issue_ahead = None
    dali_pipeline_depth = None
    nobatch_sorting = None
    num_warmups = None


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class Xavier_NX_MaxQ(Xavier_NX):
    # power settings
    soc_gpu_freq = 752250000
    soc_dla_freq = 115200000
    soc_cpu_freq = 1190400
    soc_emc_freq = 1331200000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class Orin(SingleStreamGPUBaseConfig):
    system = KnownSystem.Orin
    single_stream_expected_latency_ns = 120000000
    audio_fp16_input = None
    dali_batches_issue_ahead = None
    dali_pipeline_depth = None
    nobatch_sorting = None
    num_warmups = None
# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_80GB_PCIEX2(SingleStreamGPUBaseConfig):
    system = KnownSystem.A100_80GB_PCIex2
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = False

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
#    input_dtype: str = ''
#    map_path: str = ''
#    precision: str = ''
#    tensor_path: str = ''
#
#    # Optional fields:
#    active_sms: int = 0
#    audio_batch_size: int = 0
#    audio_buffer_num_lines: int = 0
#    audio_fp16_input: bool = False
#    buffer_manager_thread_count: int = 0
#    cache_file: str = ''
#    dali_batches_issue_ahead: int = 0
#    dali_pipeline_depth: int = 0
#    disable_encoder_plugin: bool = False
#    instance_group_count: int = 0
#    max_queue_delay_usec: int = 0
#    max_seq_length: int = 0
#    model_path: str = ''
#    nobatch_sorting: bool = False
#    noenable_audio_processing: bool = False
#    nopipelined_execution: bool = False
#    nouse_copy_kernel: bool = False
#    num_warmups: int = 0
#    performance_sample_count_override: int = 0
#    preferred_batch_size: str = ''
#    request_timeout_usec: int = 0
#    run_infer_on_copy_streams: bool = False
#    single_stream_expected_latency_ns: int = 0
#    single_stream_target_latency_percentile: float = 0.0
#    use_jemalloc: bool = False
#    use_spin_wait: bool = False
#    workspace_size: int = 0


