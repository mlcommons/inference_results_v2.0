#! /bin/bash

if [[ -z "$1" ]]; then
  echo "Usage:"
  echo "    ./scripts/$(basename "${BASH_SOURCE[0]}") npu_ids [output_dir_prefix]"
  echo
  echo "        npu_ids           - comma-separated npu numbers"
  echo "                            first npu number is used mainly"
  echo "                            Must set this value"
  echo "        output_dir        - output directory to save results, reports and profiles"
  echo "                            relative to mlperf project directory"
  echo "                            contents in out_dir can be overwritten or appended"
  echo "                            default: outputs/run_<start_time>"
  echo
  echo "Environment variables:"
  echo "        DRY_RUN                      - set this true to skip compiler, runner, mlflow"
  echo "                                       default: false"
  echo "        MLPERF_CONFIG                - mlperf config file for performance measurement"
  echo "                                       default: models/mlperf_large.conf"
  echo "        MLPERF_PROFILE_CONFIG        - mlperf config file for profiling"
  echo "                                       default: models/mlperf_small.conf"
  echo "        MLPERF_ACCURACY_CONFIG       - mlperf config file for accuracy check"
  echo "                                       default: ../loadgen/inference/mlperf.conf"
  echo "        NPU_CONF_WARBOY              - npu config file path for non-fused pe"
  echo "                                       default: warboy"
  echo "        NPU_CONF_WARBOY_2PE          - npu config file path for fused pe"
  echo "                                       default: warboy-2pe"
  echo "        DATASET_ROOT                 - set this to provide test files"
  echo "                                       accuracy test needs it"
  echo "        MLFLOW_RUN_NAME              - set this variable to send result to mlflow"
  echo "        MLPERF_RUNNER                - set this variable to use different runner"
  echo "                                       default: target/release/runner"
  echo "        DISABLE_PERF                 - default: false"
  echo "        DISABLE_ACCURACY             - default: false"
  echo "        SKIP_PROFILE                 - default: false"
  echo "        RUN_ALL                      - default: true"
  echo "        RUN_SINGLE_STREAM            - default: false"
  echo "        RUN_OFFLINE                  - default: false"
  echo "        RUN_RESNET50                 - default: false"
  echo "        RUN_RESNET50_SINGLE_STREAM   - default: false"
  echo "        RUN_RESNET50_OFFLINE         - default: false"
  echo "        RUN_SSD_SMALL                - default: false"
  echo "        RUN_SSD_SMALL_SINGLE_STREAM  - default: false"
  echo "        RUN_SSD_SMALL_OFFLINE        - default: false"
  echo "        RUN_SSD_LARGE                - default: false"
  echo "        RUN_SSD_LARGE_SINGLE_STREAM  - default: false"
  echo "        RUN_SSD_LARGE_OFFLINE        - default: false"
  echo "        REPEAT_ACCURACY_ON_NPUS      - default: false"
  echo "        REPEAT_SINGLE_STREAM_ON_NPUS - default: false"
  echo "        SCALE_OFFLINE_ON_NPUS        - default: false"
  echo "        REPEAT_OFFLINE_ON_ALL_NPUS   - default: false"
  echo "        REPEAT_OFFLINE_ON_NPUS       - default: false"
  echo "        REPEAT_OFFLINE_ON_TWO_NPUS   - default: false"
  echo "        REPEAT_OFFLINE_ON_THREE_NPUS - default: false"
  echo "        REPEAT_OFFLINE_ON_FOUR_NPUS  - default: false"
  echo "        REPEAT_ON_TASKSETS           - default: false"
  echo "        REPEAT_ON_WORKERS            - default: false"
  echo "        REPEAT_ON_POST_PROCESSORS    - default: false"
  exit 1
fi

export MLCOMMONS_USE_CACHED_GRAPH=true

npu_ids=(${1//,/ })
output_dir=${2:-"outputs/run_$(date +'%Y%m%d%H%M%S')"}
perf_conf=${MLPERF_CONFIG:-"models/mlperf_large.conf"}
profile_conf=${MLPERF_PROFILE_CONFIG:-"models/mlperf_small.conf"}
accuracy_conf=${MLPERF_ACCURACY_CONFIG:-"../loadgen/inference/mlperf.conf"}
npu_conf_warboy=${NPU_CONF_WARBOY:-"warboy"}
npu_conf_warboy_2pe=${NPU_CONF_WARBOY_2PE:-"warboy-2pe"}
runner=${MLPERF_RUNNER:-"target/release/runner"}
mlflow_tools="./mlflow_tools.py"
compiler="target/release/compile_model"
if [[ "${DRY_RUN}" == "true" ]]; then
  output_dir="${output_dir}_dry"
fi

resnet50_batch_candidates=( 8 )
ssd_small_batch_candidates=( 8 )
ssd_large_batch_candidates=( 1 )

if [[ "${REPEAT_ON_WORKERS}" == "true" ]]; then
  resnet50_workers_candidates=( 4 8 )
  ssd_small_workers_candidates=( 8 4 )
  ssd_large_workers_candidates=( 4 2 )
else
  resnet50_workers_candidates=( 4 )
  ssd_small_workers_candidates=( 8 )
  ssd_large_workers_candidates=( 4 )
fi

if [[ "${REPEAT_ON_POST_PROCESSORS}" == "true" ]]; then
  ssd_small_pp_candidates=( "cpp_par" "rust" "rust_par" )
  ssd_large_pp_candidates=( "cpp_par" "rust" "rust_par" )
else
  ssd_small_pp_candidates=( "default_pp" )
  ssd_large_pp_candidates=( "default_pp" )
fi

taskset_candidates=( "all_cpus" )
if [[ "${REPEAT_ON_TASKSETS}" ]]; then
  # Assume pid 1 uses all cpus
  all_cpus="$(taskset -p 1 | grep -E ".* ([137f]ffffffff+)$" | sed -E "s/.* ([137f]ffffffff+)$/\\1/")"
  if [[ -z "${all_cpus}" ]]; then
    all_cpus="ffffffffffffffffffffffffffffffffffffff"
  fi
  taskset_mask="ffffffff"
  while (( ${#taskset_mask} <= ${#all_cpus} )); do
    taskset_candidates+=( "${taskset_mask}" )
    taskset_mask="${taskset_mask}00000000"
  done
fi

single_npu_devnames=() # ("npu0pe0,npu0pe1" "npu1pe0,npu1pe1", ...) for offline accuracy (resnet50, ssd_small)
multi_npu_devnames=() # ("npu0pe0,npu0pe1" "npu0pe0,npu0pe1,npu1pe0,npu1pe1", ...) for offline performance (resnet50, ssd_small)
single_fused_npu_devnames=() # ("npu0pe0-1" "npu1pe0-1", ...) for single stream accuracy and performance (all)
multi_fused_npu_devnames=() # ("npu0pe0-1" "npu0pe0-1,npu1pe0-1", ...) for offline performance (ssd_large)

for npu_id in "${npu_ids[@]}"; do
  single_npu_devnames+=("npu${npu_id}pe0,npu${npu_id}pe1")
  single_fused_npu_devnames+=("npu${npu_id}pe0-1")
done

if [[ "${SCALE_OFFLINE_ON_NPUS}" == "true" ]]; then
  for single_npu_devname in "${single_npu_devnames[@]}"; do
    if [[ "${single_npu_devname}" == "${single_npu_devnames[0]}" ]]; then
      multi_npu_devname="${single_npu_devname}"
    else
      multi_npu_devname="${multi_npu_devname},${single_npu_devname}"
    fi
    multi_npu_devnames+=("${multi_npu_devname}")
  done
  for single_fused_npu_devname in "${single_fused_npu_devnames[@]}"; do
    if [[ "${single_fused_npu_devname}" == "${single_fused_npu_devnames[0]}" ]]; then
      multi_fused_npu_devname="${single_fused_npu_devname}"
    else
      multi_fused_npu_devname="${multi_fused_npu_devname},${single_fused_npu_devname}"
    fi
    multi_fused_npu_devnames+=("${multi_fused_npu_devname}")
  done
fi

if [[ "${REPEAT_OFFLINE_ON_NPUS}" == "true" ]] || [[ "${REPEAT_OFFLINE_ON_ALL_NPUS}" == "true" ]]; then
  for single_npu_devname in "${single_npu_devnames[@]}"; do
    multi_npu_devnames+=("${single_npu_devname}")
  done
  for single_fused_npu_devname in "${single_fused_npu_devnames[@]}"; do
    multi_fused_npu_devnames+=("${single_fused_npu_devname}")
  done
fi

if [[ "${REPEAT_OFFLINE_ON_TWO_NPUS}" == "true" ]] || [[ "${REPEAT_OFFLINE_ON_ALL_NPUS}" == "true" ]]; then
  len=${#single_npu_devnames[@]}
  for (( first=0; first < len-1; ++first )); do
    for (( second=first+1; second < len; ++second )); do
      multi_npu_devnames+=("${single_npu_devnames[first]},${single_npu_devnames[second]}")
      multi_fused_npu_devnames+=("${single_fused_npu_devnames[first]},${single_fused_npu_devnames[second]}")
    done
  done
fi

if [[ "${REPEAT_OFFLINE_ON_THREE_NPUS}" == "true" ]] || [[ "${REPEAT_OFFLINE_ON_ALL_NPUS}" == "true" ]]; then
  len=${#single_npu_devnames[@]}
  for (( first=0; first < len-2; ++first )); do
    for (( second=first+1; second < len-1; ++second )); do
      for (( third=second+1; third < len; ++third )); do
        multi_npu_devnames+=("${single_npu_devnames[first]},${single_npu_devnames[second]},${single_npu_devnames[third]}")
        multi_fused_npu_devnames+=("${single_fused_npu_devnames[first]},${single_fused_npu_devnames[second]},${single_fused_npu_devnames[third]}")
      done
    done
  done
fi

if [[ "${REPEAT_OFFLINE_ON_FOUR_NPUS}" == "true" ]] || [[ "${REPEAT_OFFLINE_ON_ALL_NPUS}" == "true" ]]; then
  len=${#single_npu_devnames[@]}
  for (( first=0; first < len-3; ++first )); do
    for (( second=first+1; second < len-2; ++second )); do
      for (( third=second+1; third < len-1; ++third )); do
        for (( fourth=third+1; fourth < len; ++fourth )); do
          multi_npu_devnames+=("${single_npu_devnames[first]},${single_npu_devnames[second]},${single_npu_devnames[third]},${single_npu_devnames[fourth]}")
          multi_fused_npu_devnames+=("${single_fused_npu_devnames[first]},${single_fused_npu_devnames[second]},${single_fused_npu_devnames[third]},${single_fused_npu_devnames[fourth]}")
        done
      done
    done
  done
fi

if (( ${#multi_npu_devnames[@]} == 0 )); then
  multi_npu_devnames+=("${single_npu_devnames[0]}")
fi

if (( ${#multi_fused_npu_devnames[@]} == 0 )); then
  multi_fused_npu_devnames+=("${single_fused_npu_devnames[0]}")
fi

function fake_runner() {
  (set +x; echo "fake_runner" "$@")
}

function fake_compile() {
  (set +x; echo "fake_compile" "$@")
}

function fake_mlflow_tools() {
  (set +x; echo "fake_mlflow_tools" "$@")
}

if [[ "${DRY_RUN}" == "true" ]]; then
  runner=fake_runner
  compiler=fake_compile
  mlflow_tools=fake_mlflow_tools
fi

# Change working directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p "${output_dir}/trace"
summary_file="${output_dir}/summary.txt"
touch "${summary_file}"

# Pre-compile to eliminate time between tests
NPU_GLOBAL_CONFIG_PATH="${npu_conf_warboy_2pe}" "${compiler}" --model models/resnet50_int8.onnx --remove-unlower
NPU_GLOBAL_CONFIG_PATH="${npu_conf_warboy_2pe}" "${compiler}" --model models/ssd_mobilenet_int8.onnx --remove-unlower
NPU_GLOBAL_CONFIG_PATH="${npu_conf_warboy_2pe}" "${compiler}" --model models/ssd_resnet34_int8.onnx --remove-unlower
for batch in "${resnet50_batch_candidates[@]}"; do
  NPU_GLOBAL_CONFIG_PATH="${npu_conf_warboy}" "${compiler}" --model models/resnet50_int8.onnx --batch-size "${batch}" --remove-unlower
done
for batch in "${ssd_small_batch_candidates[@]}"; do
  NPU_GLOBAL_CONFIG_PATH="${npu_conf_warboy}" "${compiler}" --model models/ssd_mobilenet_int8.onnx --batch-size "${batch}" --remove-unlower
done
for batch in "${ssd_large_batch_candidates[@]}"; do
  NPU_GLOBAL_CONFIG_PATH="${npu_conf_warboy_2pe}" "${compiler}" --model models/ssd_resnet34_int8.onnx --batch-size "${batch}" --remove-unlower
done

function accuracy_resnet50() {
  echo "Calculating resnet50 accuracy..."
  local result=$(python3 ../loadgen/inference/vision/classification_and_detection/tools/accuracy-imagenet.py \
    --mlperf-accuracy-file mlperf_resnet50_out/mlperf_log_accuracy.json \
    --imagenet-val-file models/val_map.txt)
  local baseline=$(tail results/resnet50/accuracy.txt)
  if [[ "${baseline}" == "${result}" ]]; then
    echo "Accuracy unchanged: ${result}" | tee -a "${summary_file}"
  else
    echo "Accuracy CHANGED: ${baseline} -> ${result}" | tee -a "${summary_file}"
  fi
}

function accuracy_ssd_small() {
  echo "Calculating ssd_small accuracy..."
  local result=$(python3 ../loadgen/inference/vision/classification_and_detection/tools/accuracy-coco.py \
    --mlperf-accuracy-file mlperf_ssd_small_out/mlperf_log_accuracy.json \
    --coco-dir models | tail -n 1)
  local baseline=$(tail -n 1 results/ssd_small/accuracy.txt)
  if [[ "${baseline}" == "${result}" ]]; then
    echo "Accuracy unchanged: ${result}" | tee -a "${summary_file}"
  else
    echo "Accuracy CHANGED: ${baseline} -> ${result}" | tee -a "${summary_file}"
  fi
}

function accuracy_ssd_large() {
  echo "Calculating ssd_large accuracy..."
  local result=$(python3 ../loadgen/inference/vision/classification_and_detection/tools/accuracy-coco.py \
    --mlperf-accuracy-file mlperf_ssd_large_out/mlperf_log_accuracy.json \
    --coco-dir models --use-inv-map | tail -n 1)
  local baseline=$(tail -n 1 results/ssd_large/accuracy.txt)
  if [[ "${baseline}" == "${result}" ]]; then
    echo "Accuracy unchanged: ${result}" | tee -a "${summary_file}"
  else
    echo "Accuracy CHANGED: ${baseline} -> ${result}" | tee -a "${summary_file}"
  fi
}

function accuracy() {
  local model=$1
  if [[ "${model}" == "resnet50" ]]; then
    accuracy_resnet50
  elif [[ "${model}" == "ssd_small" ]]; then
    accuracy_ssd_small
  elif [[ "${model}" == "ssd_large" ]]; then
    accuracy_ssd_large
  fi
}

function write_md_perf_header() {
  local model=$1
  local scenario=$2
  local filename="${output_dir}/${model}_${scenario}.md"
  if [[ "${DRY_RUN}" == "true" ]]; then
    return
  fi
  touch "${filename}"
  if [[ "${scenario}" == "SingleStream" ]]; then
    { echo; \
    echo "| model | 90th %ile latency (ns)  | NPU_DEVNAME | PP | taskset| is_valid |"; \
    echo "| --- | --- | --- | --- | --- | --- |"; } >> "${filename}"
  elif [[ "${scenario}" == "Offline" ]]; then
    { echo; \
    echo "| model | Samples/sec  | NPU_DEVNAME | batch_size | PP | workers | taskset| is_valid |"; \
    echo "| --- | --- | --- | --- | --- | --- | --- | --- |"; } >> "${filename}"
  fi
}

write_md_perf_header resnet50 SingleStream
write_md_perf_header resnet50 Offline
write_md_perf_header ssd_small SingleStream
write_md_perf_header ssd_small Offline
write_md_perf_header ssd_large SingleStream
write_md_perf_header ssd_large Offline

function md_perf_out() {
  local model=$1
  local mode=$2
  local scenario=$3
  local npu_devname=$4
  local batch_size=$5
  local workers=$6
  local post_processor=$7
  local profile=$8
  local taskset_mask=$9

  if [[ "${DRY_RUN}" == "true" ]]; then
    return
  fi

  if [[ "${scenario}" == "SingleStream" ]]; then
    local result=$(grep -E "90th percentile latency \(ns\)" "mlperf_${model}_out/mlperf_log_summary.txt" | awk '{print $6}')
  elif [[ "${scenario}" == "Offline" ]]; then
    local result=$(grep -E "Samples per second" "mlperf_${model}_out/mlperf_log_summary.txt" | awk '{print $4}')
  fi
  local is_valid=$(grep -E "Result is : " "mlperf_${model}_out/mlperf_log_summary.txt" | awk '{print $4}')

  if [[ "${scenario}" == "SingleStream" ]]; then
    echo "| ${model} | ${result}  | ${npu_devname} | ${post_processor} | ${taskset_mask}| ${is_valid} |" >> "${output_dir}/${model}_${scenario}.md"
  elif [[ "${scenario}" == "Offline" ]]; then
    echo "| ${model} | ${result}  | ${npu_devname} | ${batch_size} | ${post_processor} | ${workers} | ${taskset_mask}| ${is_valid} |" >> "${output_dir}/${model}_${scenario}.md"
  fi
}

function input_path() {
  local model=$1
  if [[ "${model}" == "resnet50" ]]; then
    echo "${DATASET_ROOT}/imagenet-golden/raw/"
  elif [[ "${model}" == "ssd_small" ]]; then
    echo "${DATASET_ROOT}/coco-300-golden/raw/"
  elif [[ "${model}" == "ssd_large" ]]; then
    echo "${DATASET_ROOT}/coco-1200-golden/raw/"
  fi
}

function npu_count() {
  local devname="${1//pe0-1/}"
  devname="${devname//pe0/}"
  devname="${devname//pe1/}"
  devname="${devname//npu/}"
  local current_npu_ids=(${devname//,/ })
  for current_npu_id in "${current_npu_ids[@]}"; do
    echo "${current_npu_id}"
  done | sort | uniq | wc -l | awk '{$1=$1;print}'
}

function is_multi_npu() {
  if [[ "$(npu_count "$1")" == "1" ]]; then
    echo "false"
  else
    echo "true"
  fi
}

function mlflow() {
  local model=$1
  local mode=$2
  local scenario=$3
  local npu_devname=$4
  local batch_size=$5
  local workers=$6
  local post_processor=$7
  local profile=$8
  local taskset_mask=$9
  local profiler_path=${10}

  if [[ -z "${MLFLOW_RUN_NAME}" ]]; then
    echo "Skip mlflow upload because MLFLOW_RUN_NAME is not set"
    return
  fi
  cd ..
  local extra_args=()
  if [[ "${post_processor}" != "default_pp" ]]; then
    extra_args+=( "--additional-param" "post-processor=${post_processor}" )
  fi
  if [[ -n "${taskset_mask}" ]]; then
    extra_args+=( "--additional-tag" "taskset_mask=${taskset_mask}" )
  fi
  if [[ -n "${profiler_path}" ]]; then
    extra_args+=( "--artifact-path" "${profiler_path}" "--additional-tag" "profiling=true" )
  fi
  extra_args+=( "--additional-tag" "npu_count=$(npu_count "${npu_devname}")" )
  if [[ "$(is_multi_npu "${npu_devname}")" == "true" ]]; then
    extra_args+=( "--additional-tag" "multi_npu=true" )
  fi
  set -x
  "${mlflow_tools}" submit --run-name "${MLFLOW_RUN_NAME}" --model "${model}" --scenario "${scenario}" \
    --npu-devname "${npu_devname}" --batch-size "${batch_size}" --workers "${workers}" \
    "${extra_args[@]}" --additional-tag "exhaustive_run=true"
  set +x
  cd mlperf
}

function run() {
  if (( $# < 9 )); then
    echo "Error: num args $#, expected 9 in run: $*" | tee -a "${summary_file}"
    return
  fi
  local model=$1
  local mode=$2
  local scenario=$3
  local npu_devname=$4
  local batch_size=$5
  local workers=$6
  local post_processor=$7
  local profile=$8
  local taskset_mask=$9

  local run_id="${model}_${mode}_${scenario}_${npu_devname//,/_}_b${batch_size}"
  local conf_file="${perf_conf}"
  local npu_global_config_path="${npu_conf_warboy}"
  local extra_args=()

  echo "---------- begin @ $(date) ----------"
  if [[ "${npu_devname}" == *"pe0-1"* ]]; then
    npu_global_config_path="${npu_conf_warboy_2pe}"
  fi
  if [[ "${mode}" == "AccuracyOnly" ]]; then
    conf_file="${accuracy_conf}"
    profile="false"
  fi
  if [[ "${post_processor}" != "default_pp" ]]; then
    run_id="${run_id}_${post_processor}"
    extra_args+=( "--post-processor" "${post_processor}" )
  fi
  run_id="${run_id}_w${workers}"
  if [[ -n "${DATASET_ROOT}" ]]; then
    extra_args+=( "--input" "$(input_path "${model}")" )
  fi
  taskset_args=()
  if [[ "${taskset_mask}" != "all_cpus" ]]; then
    if [[ "${DRY_RUN}" != "true" ]]; then
      taskset_args+=( "taskset" "${taskset_mask}" )
    else
      echo "Taskset: ${taskset_mask}"
    fi
    run_id="${run_id}_${taskset_mask}"
  fi
  run_id="${run_id}_$(basename "${runner}")"
  if [[ "${profile}" == "true" ]]; then
    echo "Profile: enabled"
    export RUST_LOG=info
    export NPU_PROFILER_PATH="${output_dir}/trace/npu_${run_id}.json"
    export NUX_PROFILER_PATH="${output_dir}/trace/nux_${run_id}.json"
    run_id="${run_id}_profiled"
    conf_file="${profile_conf}"
  else
    echo "Profile: disabled"
    unset RUST_LOG
    unset NPU_PROFILER_PATH
    unset NUX_PROFILER_PATH
  fi
  set -x
  NPU_DEVNAME="${npu_devname}" NPU_GLOBAL_CONFIG_PATH="${npu_global_config_path}" "${taskset_args[@]}" "${runner}" \
    --config "${conf_file}" --model "${model}" --scenario "${scenario}" --mode "${mode}" \
    --batch-size "${batch_size}" --workers "${workers}" \
    "${extra_args[@]}"
  ret_code=$?
  set +x
  echo "- ${run_id}" >> "${summary_file}"
  if [[ "${ret_code}" == "0" ]]; then
    cat "mlperf_${model}_out/mlperf_log_summary.txt"
    { grep -E "(Samples per second|90th percentile latency \(ns\))" "mlperf_${model}_out/mlperf_log_summary.txt"; \
    grep -E "Result is : " "mlperf_${model}_out/mlperf_log_summary.txt";
    grep -E "satisfied" "mlperf_${model}_out/mlperf_log_summary.txt"; } >> "${summary_file}"
    if [[ "${mode}" == "AccuracyOnly" ]]; then
      accuracy "${model}"
    else
      if [[ "${profile}" != "true" ]]; then
        md_perf_out "$@"
      fi
      mlflow "$@" "${NUX_PROFILER_PATH}"
    fi
  else
    echo "FAILED with return code ${ret_code}" | tee -a "${summary_file}"
  fi
  echo >> "${summary_file}"
  echo "---------- end @ $(date) ----------"
  echo
}

function run_performance_set() {
  local model=$1
  local scenario=$2
  local npu_devname=$3
  local batch_size=$4
  local workers=$5
  local post_processor=$6
  if [[ "${DISABLE_PERF}" == "true" ]]; then
    return
  fi
  for taskset_mask in "${taskset_candidates[@]}"; do
    run "${model}" PerformanceOnly "${scenario}" "${npu_devname}" "${batch_size}" "${workers}" "${post_processor}" "false" "${taskset_mask}"
    if [[ "${SKIP_PROFILE}" != "true" ]]; then
      run "${model}" PerformanceOnly "${scenario}" "${npu_devname}" "${batch_size}" "${workers}" "${post_processor}" "true" "${taskset_mask}"
    fi
  done
}

function run_accuracy() {
  local model=$1
  local scenario=$2
  local npu_devname=$3
  local batch_size=$4
  local workers=$5
  local post_processor=$6
  if [[ "${DISABLE_ACCURACY}" == "true" ]]; then
    return
  fi
  if [[ -z "${DATASET_ROOT}" ]]; then
    return
  fi
  run "${model}" AccuracyOnly "${scenario}" "${npu_devname}" "${batch_size}" "${workers}" "${post_processor}" "false" "${taskset_mask}"
}

function run_resnet50_single_stream() {
  local do_perf="true"
  local do_accuracy="true"
  for npu_devname in "${single_fused_npu_devnames[@]}"; do
    if [[ "${do_perf}" == "true" ]]; then
      run_performance_set "resnet50" "SingleStream" "${npu_devname}" 1 1 "default_pp"
    fi
    if [[ "${do_accuracy}" == "true" ]]; then
      run_accuracy "resnet50" "SingleStream" "${npu_devname}" 1 1 "default_pp"
    fi
    if [[ "${REPEAT_SINGLE_STREAM_ON_NPUS}" != "true" ]]; then
      do_perf="false"
    fi
    if [[ "${REPEAT_ACCURACY_ON_NPUS}" != "true" ]]; then
      do_accuracy="false"
    fi
  done
}

function run_ssd_small_single_stream() {
  for post_processor in "${ssd_small_pp_candidates[@]}"; do
    local do_perf="true"
    local do_accuracy="true"
    for npu_devname in "${single_fused_npu_devnames[@]}"; do
      if [[ "${do_perf}" == "true" ]]; then
        run_performance_set "ssd_small" "SingleStream" "${npu_devname}" 1 1 "${post_processor}"
      fi
      if [[ "${do_accuracy}" == "true" ]]; then
        run_accuracy "ssd_small" "SingleStream" "${npu_devname}" 1 1 "${post_processor}"
      fi
      if [[ "${REPEAT_SINGLE_STREAM_ON_NPUS}" != "true" ]]; then
        do_perf="false"
      fi
      if [[ "${REPEAT_ACCURACY_ON_NPUS}" != "true" ]]; then
        do_accuracy="false"
      fi
    done
  done
}

function run_ssd_large_single_stream() {
  for post_processor in "${ssd_large_pp_candidates[@]}"; do
    local do_perf="true"
    local do_accuracy="true"
    for npu_devname in "${single_fused_npu_devnames[@]}"; do
      if [[ "${do_perf}" == "true" ]]; then
        run_performance_set "ssd_large" "SingleStream" "${npu_devname}" 1 1 "${post_processor}"
      fi
      if [[ "${do_accuracy}" == "true" ]]; then
        run_accuracy "ssd_large" "SingleStream" "${npu_devname}" 1 1 "${post_processor}"
      fi
      if [[ "${REPEAT_SINGLE_STREAM_ON_NPUS}" != "true" ]]; then
        do_perf="false"
      fi
      if [[ "${REPEAT_ACCURACY_ON_NPUS}" != "true" ]]; then
        do_accuracy="false"
      fi
    done
  done
}

function run_resnet50_offline() {
  for batch_size in "${resnet50_batch_candidates[@]}"; do
    for npu_devname in "${multi_npu_devnames[@]}"; do
      for workers in "${resnet50_workers_candidates[@]}"; do
        run_performance_set "resnet50" "Offline" "${npu_devname}" "${batch_size}" "${workers}" "default_pp"
      done
    done
    for npu_devname in "${single_npu_devnames[@]}"; do
      run_accuracy "resnet50" "Offline" "${npu_devname}" "${batch_size}" 8 "default_pp"
      if [[ "${REPEAT_ACCURACY_ON_NPUS}" != "true" ]]; then
        break
      fi
    done
  done
}

function run_ssd_small_offline() {
  for batch_size in "${ssd_small_batch_candidates[@]}"; do
    for post_processor in "${ssd_small_pp_candidates[@]}"; do
      for npu_devname in "${multi_npu_devnames[@]}"; do
        for workers in "${ssd_small_workers_candidates[@]}"; do
          run_performance_set "ssd_small" "Offline" "${npu_devname}" "${batch_size}" "${workers}" "${post_processor}"
        done
      done
      for npu_devname in "${single_npu_devnames[@]}"; do
        run_accuracy "ssd_small" "Offline" "${npu_devname}" "${batch_size}" 8 "${post_processor}"
        if [[ "${REPEAT_ACCURACY_ON_NPUS}" != "true" ]]; then
          break
        fi
      done
    done
  done
}

function run_ssd_large_offline() {
  for batch_size in "${ssd_large_batch_candidates[@]}"; do
    for post_processor in "${ssd_large_pp_candidates[@]}"; do
      for npu_devname in "${multi_fused_npu_devnames[@]}"; do
        for workers in "${ssd_large_workers_candidates[@]}"; do
          run_performance_set "ssd_large" "Offline" "${npu_devname}" "${batch_size}" "${workers}" "${post_processor}"
        done
      done
      for npu_devname in "${single_fused_npu_devnames[@]}"; do
        run_accuracy "ssd_large" "Offline" "${npu_devname}" "${batch_size}" 8 "${post_processor}"
        if [[ "${REPEAT_ACCURACY_ON_NPUS}" != "true" ]]; then
          break
        fi
      done
    done
  done
}

echo "====== started at $(date) =====" >> "${summary_file}"
if [[ "${RUN_ALL}" != "false" ]] || [[ "${RUN_RESNET50_SINGLE_STREAM}" == "true" ]] || [[ "${RUN_RESNET50}" == "true" ]] || [[ "${RUN_SINGLE_STREAM}" == "true" ]]; then
  run_resnet50_single_stream
fi
if [[ "${RUN_ALL}" != "false" ]] || [[ "${RUN_RESNET50_OFFLINE}" == "true" ]] || [[ "${RUN_RESNET50}" == "true" ]] || [[ "${RUN_OFFLINE}" == "true" ]]; then
  run_resnet50_offline
fi
if [[ "${RUN_ALL}" != "false" ]] || [[ "${RUN_SSD_SMALL_SINGLE_STREAM}" == "true" ]] || [[ "${RUN_SSD_SMALL}" == "true" ]] || [[ "${RUN_SINGLE_STREAM}" == "true" ]]; then
  run_ssd_small_single_stream
fi
if [[ "${RUN_ALL}" != "false" ]] || [[ "${RUN_SSD_SMALL_OFFLINE}" == "true" ]] || [[ "${RUN_SSD_SMALL}" == "true" ]] || [[ "${RUN_OFFLINE}" == "true" ]]; then
  run_ssd_small_offline
fi
if [[ "${RUN_ALL}" != "false" ]] || [[ "${RUN_SSD_LARGE_SINGLE_STREAM}" == "true" ]] || [[ "${RUN_SSD_LARGE}" == "true" ]] || [[ "${RUN_SINGLE_STREAM}" == "true" ]]; then
  run_ssd_large_single_stream
fi
if [[ "${RUN_ALL}" != "false" ]] || [[ "${RUN_SSD_LARGE_OFFLINE}" == "true" ]] || [[ "${RUN_SSD_LARGE}" == "true" ]] || [[ "${RUN_OFFLINE}" == "true" ]]; then
  run_ssd_large_offline
fi
echo "====== finished at $(date) =====" >> "${summary_file}"
echo "All data written to ${output_dir}"