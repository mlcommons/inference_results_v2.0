# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2460M1_A30x4(ServerGPUBaseConfig):
    system = KnownSystem.GX2460M1_A30x4

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    input_dtype: str = 'int32'
    precision: str = 'int8'
    tensor_path: str = '${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy'
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = False
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    input_format = "linear"
    use_graphs = True
    active_sms = 60
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 32
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 6052
    soft_drop = 0.993
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GX2460M1_A30x4_HighAccuracy(GX2460M1_A30x4):
    precision = "fp16"
    server_target_qps = 2627

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class GX2460M1_A30x4_Triton(GX2460M1_A30x4):
    use_triton = True

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    input_dtype: str = ''
    precision: str = ''
    tensor_path: str = ''

    # Optional fields:
    active_sms: int = 0
    batch_triton_requests: bool = False
    bert_opt_seqlen: int = 0
    buffer_manager_thread_count: int = 0
    cache_file: str = ''
    coalesced_tensor: bool = False
    deque_timeout_usec: int = 0
    gather_kernel_buffer_threshold: int = 0
    graph_specs: str = ''
    graphs_max_seqlen: int = 0
    instance_group_count: int = 0
    max_queue_delay_usec: int = 0
    model_path: str = ''
    num_concurrent_batchers: int = 0
    num_concurrent_issuers: int = 0
    output_pinned_memory: bool = False
    performance_sample_count_override: int = 0
    preferred_batch_size: str = ''
    request_timeout_usec: int = 0
    run_infer_on_copy_streams: bool = False
    schedule_rng_seed: int = 0
    server_num_issue_query_threads: int = 0
    server_target_latency_ns: int = 0
    server_target_latency_percentile: float = 0.0
    server_target_qps: int = 0
    soft_drop: float = 0.0
    use_concurrent_harness: bool = False
    use_jemalloc: bool = False
    use_spin_wait: bool = False
    workspace_size: int = 0


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GX2460M1_A30x4_HighAccuracy_Triton(GX2460M1_A30x4_HighAccuracy):
    use_triton = True


