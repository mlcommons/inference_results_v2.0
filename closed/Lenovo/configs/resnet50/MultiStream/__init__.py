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
from configs.resnet50 import GPUBaseConfig


class MultiStreamGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.MultiStream
    gpu_batch_size = 8
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    multi_stream_samples_per_query = 8
    multi_stream_target_latency_percentile = 99
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex1(MultiStreamGPUBaseConfig):
    system = KnownSystem.A100_PCIe_40GBx1
    multi_stream_expected_latency_ns = 830000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex1_Triton(A100_PCIex1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1(MultiStreamGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBx1
    multi_stream_expected_latency_ns = 830000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1_Triton(A100_PCIe_80GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1(MultiStreamGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_ARMx1
    multi_stream_expected_latency_ns = 830000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1_Triton(A100_PCIe_80GB_aarch64x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1(MultiStreamGPUBaseConfig):
    system = KnownSystem.A100_PCIe_40GB_ARMx1
    multi_stream_expected_latency_ns = 830000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1_Triton(A100_PCIe_aarch64x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb(MultiStreamGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_MIG_1x1g_10gb
    multi_stream_expected_latency_ns = 2160000


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_Hetero(A100_PCIe_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_Triton(A100_PCIe_80GB_MIG_1x1g10gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_56x1g10gb_Triton(MultiStreamGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_MIG_56x1g_10gb
    use_triton = True
    multi_stream_expected_latency_ns = 2160000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb(MultiStreamGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_MIG_1x1g_10gb
    start_from_device = True
    multi_stream_expected_latency_ns = 2100000


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Triton(A100_SXM_80GB_MIG_1x1g10gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_56x1g10gb_Triton(MultiStreamGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_MIG_56x1g_10gb
    start_from_device = True
    use_triton = True
    multi_stream_expected_latency_ns = 2100000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1(MultiStreamGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx1
    start_from_device = True
    multi_stream_expected_latency_ns = 693000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1_Triton(A100_SXM_80GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x1(MultiStreamGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_ARMx1
    multi_stream_expected_latency_ns = 6430000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x1_Triton(A100_SXM_80GB_aarch64x1):
    use_triton = True
    multi_stream_expected_latency_ns = 1000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb(MultiStreamGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_ARM_MIG_1x1g_10gb
    multi_stream_expected_latency_ns = 2100000


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Triton(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb(MultiStreamGPUBaseConfig):
    system = KnownSystem.A100_SXM4_40GB_MIG_1x1g_5gb
    start_from_device = True
    multi_stream_expected_latency_ns = 670000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb_Triton(A100_SXM4_40GB_MIG_1x1g5gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1(MultiStreamGPUBaseConfig):
    system = KnownSystem.A100_SXM4_40GBx1
    start_from_device = True
    multi_stream_expected_latency_ns = 670000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1_Triton(A100_SXM4_40GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1(MultiStreamGPUBaseConfig):
    system = KnownSystem.A2x1
    multi_stream_expected_latency_ns = 5840000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1_Triton(A2x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(MultiStreamGPUBaseConfig):
    system = KnownSystem.A30_MIG_1x1g_6gb
    multi_stream_expected_latency_ns = 5726152


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    multi_stream_expected_latency_ns = 5968320


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(A30_MIG_1x1g6gb):
    use_graphs = False
    use_triton = True
    multi_stream_expected_latency_ns = 6889088


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_32x1g6gb_Triton(MultiStreamGPUBaseConfig):
    system = KnownSystem.A30_MIG_32x1g_6gb
    use_graphs = False
    use_triton = True
    multi_stream_expected_latency_ns = 6889088


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(MultiStreamGPUBaseConfig):
    system = KnownSystem.A30x1
    multi_stream_expected_latency_ns = 960000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_Triton(A30x1):
    use_graphs = False
    use_triton = True
    multi_stream_expected_latency_ns = 1330000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class AGX_Xavier(MultiStreamGPUBaseConfig):
    system = KnownSystem.AGX_Xavier
    gpu_copy_streams = 2
    multi_stream_expected_latency_ns = 18184000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class AGX_Xavier_MaxQ(AGX_Xavier):
    # power settings
    soc_gpu_freq = 828750000
    soc_dla_freq = 115200000
    soc_cpu_freq = 1190400
    soc_emc_freq = 1600000000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class AGX_Xavier_Triton(AGX_Xavier):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1(MultiStreamGPUBaseConfig):
    system = KnownSystem.T4x1
    multi_stream_expected_latency_ns = 7973184


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1_Triton(T4x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class Xavier_NX(MultiStreamGPUBaseConfig):
    system = KnownSystem.Xavier_NX
    gpu_copy_streams = 2
    multi_stream_expected_latency_ns = 36368000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class Xavier_NX_MaxQ(Xavier_NX):
    # power settings
    soc_gpu_freq = 803250000
    soc_dla_freq = 115200000
    soc_cpu_freq = 1190400
    soc_emc_freq = 1331200000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Xavier_NX_Triton(Xavier_NX):
    use_triton = True


# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *



@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_80GB_PCIEX2(MultiStreamGPUBaseConfig):
    system = KnownSystem.A100_80GB_PCIex2
    multi_stream_expected_latency_ns = 830000

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
#    input_dtype: str = ''
#    map_path: str = ''
#    multi_stream_expected_latency_ns: int = 0
#    multi_stream_samples_per_query: int = 0
#    multi_stream_target_latency_percentile: float = 0.0
#    precision: str = ''
#    tensor_path: str = ''
#
#    # Optional fields:
#    active_sms: int = 0
#    assume_contiguous: bool = False
#    buffer_manager_thread_count: int = 0
#    cache_file: str = ''
#    complete_threads: int = 0
#    deque_timeout_usec: int = 0
#    instance_group_count: int = 0
#    max_queue_delay_usec: int = 0
#    model_path: str = ''
#    performance_sample_count_override: int = 0
#    preferred_batch_size: str = ''
#    request_timeout_usec: int = 0
#    run_infer_on_copy_streams: bool = False
#    use_batcher_thread_per_device: bool = False
#    use_deque_limit: bool = False
#    use_jemalloc: bool = False
#    use_same_context: bool = False
#    use_spin_wait: bool = False
#    warmup_duration: float = 0.0
#    workspace_size: int = 0
#

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_80GB_PCIEX2_Triton(A100_80GB_PCIEX2):
    use_triton = True
    multi_stream_expected_latency_ns = 830000

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
#    input_dtype: str = ''
#    map_path: str = ''
#    multi_stream_expected_latency_ns: int = 0
#    multi_stream_samples_per_query: int = 0
#    multi_stream_target_latency_percentile: float = 0.0
#    precision: str = ''
#    tensor_path: str = ''
#
#    # Optional fields:
#    active_sms: int = 0
#    assume_contiguous: bool = False
#    batch_triton_requests: bool = False
#    buffer_manager_thread_count: int = 0
#    cache_file: str = ''
#    complete_threads: int = 0
#    deque_timeout_usec: int = 0
#    gather_kernel_buffer_threshold: int = 0
#    instance_group_count: int = 0
#    max_queue_delay_usec: int = 0
#    model_path: str = ''
#    num_concurrent_batchers: int = 0
#    num_concurrent_issuers: int = 0
#    output_pinned_memory: bool = False
#    performance_sample_count_override: int = 0
#    preferred_batch_size: str = ''
#    request_timeout_usec: int = 0
#    run_infer_on_copy_streams: bool = False
#    use_batcher_thread_per_device: bool = False
#    use_concurrent_harness: bool = False
#    use_deque_limit: bool = False
#    use_jemalloc: bool = False
#    use_same_context: bool = False
#    use_spin_wait: bool = False
#    warmup_duration: float = 0.0
#    workspace_size: int = 0


