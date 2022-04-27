#![allow(unused_extern_crates)]
extern crate openmp_sys;

use std::convert::TryInto;
use std::ffi::CString;
use std::fs::File;
use std::mem;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cpp::cpp;
use itertools::Itertools;
use npu_ir::dfg::Graph;
use nux::blocking::single_threaded::BlockingSession;
use rayon::prelude::*;

use loadgen::mlperf::{
    self, QuerySample, QuerySampleIndex, QuerySampleLatency, QuerySampleLibrary, SystemUnderTest,
    TestMode, TestScenario,
};
use npu_compiler::compile::GraphLoweringScheme;
use npu_executor::GraphExecutor;

use nux::async_nux::session::{self, Input, RawSession};
use nux::{HasModel, SessionOptions, TensorArray};

use crate::utils::{
    compile, compile_with_config_from_device, prepare_log_settings, prepare_test_settings,
    run_warmup, select_fused_device, select_non_fused_devices, uninitialized_vec, BoundingBox,
    CenteredBox, DetectionResult, DetectionResults, ImageStore, LoweredShape, PostProcessor,
    Postprocess, RunOptions, Shape,
};

const FEATURE_MAP_SHAPES: [usize; 6] = [19, 10, 5, 3, 2, 1];
const ANCHOR_STRIDES: [usize; 6] = [19 * 19, 10 * 10, 5 * 5, 3 * 3, 2 * 2, 1];
const NUM_ANCHORS: [usize; 6] = [3, 6, 6, 6, 6, 6];
const DATA_PATH: &str = "coco-300-golden/raw/";
const META_PATH: &str = "models/annotations/instances_val2017.json";
const TOTAL_SAMPLES: usize = 5000;

// 19x19x3 + 10x10x6 + 5x5x6 + 3x3x6 + 2x2x6 + 1x1x6
const CHANNEL_COUNT: usize = 1917;
const NUM_CLASSES: usize = 91;
const SIZE_OF_F32: usize = mem::size_of::<f32>();
// 0~5 scores 6~11 boxes
const NUM_OUTPUTS: usize = 12;
const SCALE_XY: f32 = 0.1;
const SCALE_WH: f32 = 0.2;
const SCORE_THRESHOLD: f32 = 0.3f32;
const NMS_THRESHOLD: f32 = 0.6f32;

const WARMUP_RUNS: usize = 10;

cpp! {{
    #include "cpp/unlower.h"
    #include "cpp/ssd_small.h"
    #include "bindings.h"
}}
/// Furiosa System Under Test
struct BlockingFuriosaSUT {
    name: CString,
    images: ImageStore,
    image_map: Vec<usize>,
    total_sample_count: usize,
    batch_size: usize,
    image_path: Option<PathBuf>,
    session: BlockingSession,
    postprocessor: CppPostprocessor,
    inputs: TensorArray,
    outputs: TensorArray,
}

impl SystemUnderTest for BlockingFuriosaSUT {
    fn name(&self) -> &std::ffi::CStr {
        self.name.as_c_str()
    }

    #[tracing::instrument(
        target = "chrome_layer",
        fields(name = "IssueQuery", cat = "Mlperf"),
        skip(self)
    )]
    fn issue_query(&mut self, samples: &[QuerySample]) {
        for sample in samples {
            self.inputs[0].buffer = self.images.buffer(sample.index);

            self.session.run2(&self.inputs, &mut self.outputs).unwrap();

            let mut result_data =
                self.postprocessor.postprocess_impl2(sample.index as f32, &self.outputs);

            if sample.id != 0 {
                mlperf::query_samples_complete(&mut [result_data.to_query_response(sample.id)]);
            }
        }
    }

    fn flush_queries(&mut self) {}

    fn report_latency_results(&mut self, _latencies_ns: &[QuerySampleLatency]) {}
}

impl BlockingFuriosaSUT {
    fn new(
        name: CString,
        total_sample_count: usize,
        scenario: TestScenario,
        image_path: Option<PathBuf>,
        mut session_options: SessionOptions,
        run_options: RunOptions,
    ) -> eyre::Result<Self> {
        let reader = File::open(META_PATH)?;
        let meta_data: serde_json::Value =
            serde_json::from_reader(reader).expect("JSON is not well-formatted");
        assert!(meta_data.is_object());
        let mut image_map: Vec<usize> = Vec::with_capacity(TOTAL_SAMPLES);
        let images = meta_data.get("images").unwrap().as_array().unwrap();
        for image in images {
            image_map.push(image.get("id").unwrap().as_u64().unwrap().try_into()?);
        }

        let devices = if run_options.optimal_pe_fuse { select_devices(scenario)? } else { vec![] };
        let (preprocessor, binary, main) =
            compile_ssd_small(&devices, run_options.batch_size, scenario)?;
        let postprocessor = CppPostprocessor::new(&main);

        if !devices.is_empty() {
            session_options.device(devices.join(","));
        }

        let session = BlockingSession::with(&binary, &session_options)?;
        let inputs = session.allocate_input_tensors().unwrap();
        let outputs = session.allocate_output_tensors().unwrap();

        Ok(Self {
            name,
            total_sample_count,
            batch_size: run_options.batch_size,
            images: ImageStore::new(preprocessor),
            image_map,
            image_path,
            session,
            postprocessor,
            inputs,
            outputs,
        })
    }

    fn warmup(&mut self, samples: &[QuerySampleIndex]) {
        let index = samples[samples.len() / 2];
        for _ in 0..100 {
            self.issue_query(&[QuerySample { id: 0, index }]);
        }
    }
}

impl QuerySampleLibrary for BlockingFuriosaSUT {
    fn name(&self) -> &std::ffi::CStr {
        self.name.as_c_str()
    }

    fn total_sample_count(&self) -> usize {
        TOTAL_SAMPLES
    }

    fn performance_sample_count(&self) -> usize {
        256
    }

    #[tracing::instrument(
        target = "chrome_layer",
        fields(name = "LoadSamplesToRam", cat = "Mlperf"),
        skip(self, samples)
    )]
    fn load_samples_to_ram(&mut self, samples: &[QuerySampleIndex]) {
        for &index in samples {
            // In test environment, we may have only small number of samples with low index numbers.
            // To load samples up to performance sample count and to measure the memory and cache performance,
            // reload same files to another buffer even if self.total_sample_count < TOTAL_SAMPLES.
            // Accuracy test will fail in that case.
            let wrapped_index = index % self.total_sample_count;
            let data_path =
                if let Some(path) = &self.image_path { path.to_str().unwrap() } else { DATA_PATH };
            let filename = format!("{}{:012}.png.raw", data_path, self.image_map[wrapped_index]);
            self.images.load_image(index, filename, self.batch_size);
        }
        self.warmup(samples);
    }

    #[tracing::instrument(
        target = "chrome_layer",
        fields(name = "UnloadSamplesFromRam", cat = "Mlperf"),
        skip(self, samples)
    )]
    fn unload_samples_from_ram(&mut self, samples: &[QuerySampleIndex]) {
        for &index in samples {
            self.images.unload_image(index);
        }
    }
}

pub fn run_with_blocking_session(
    image_path: Option<PathBuf>,
    scenario: TestScenario,
    mode: TestMode,
    config_path: impl AsRef<Path>,
    session_options: SessionOptions,
    run_options: RunOptions,
) {
    let total_sample_count = if image_path.is_none() { 10 } else { TOTAL_SAMPLES };
    let model = CString::new("ssd-mobilenet").unwrap();
    let test_settings = prepare_test_settings(config_path, &model, scenario, mode);
    let log_settings = prepare_log_settings("mlperf_ssd_small_out");

    let mut sut = BlockingFuriosaSUT::new(
        model,
        total_sample_count,
        scenario,
        image_path,
        session_options,
        run_options,
    )
    .unwrap();

    {
        let span = tracing::info_span!(target : "chrome_layer", "mlperf::start_test_ex", name = "StartTestEx", cat = "Mlperf");
        let _enter = span.enter();
        mlperf::start_test_ex(&mut sut, &test_settings, &log_settings);
    }
    {
        let span = tracing::info_span!(target : "chrome_layer", "FuriosaSUT::shutdown", name = "Shutdown", cat = "Mlperf");
        let _enter = span.enter();
    }
}

/// Furiosa System Under Test
struct FuriosaSUT {
    runtime: Arc<tokio::runtime::Runtime>,
    name: CString,
    images: ImageStore,
    postprocessor: Arc<dyn Postprocess + Sync + Send + 'static>,
    image_map: Vec<usize>,
    total_sample_count: usize,
    batch_size: usize,
    image_path: Option<PathBuf>,
    scenario: TestScenario,
    raw_session: Option<RawSession>,
    workload_tx: Option<crossbeam::channel::Sender<(Vec<Input>, Vec<QuerySample>)>>,
    worker_threads: Vec<Option<std::thread::JoinHandle<()>>>,
}

impl SystemUnderTest for FuriosaSUT {
    fn name(&self) -> &std::ffi::CStr {
        self.name.as_c_str()
    }

    #[tracing::instrument(
        target = "chrome_layer",
        fields(name = "IssueQuery", cat = "Mlperf"),
        skip(self)
    )]
    fn issue_query(&mut self, samples: &[QuerySample]) {
        assert!(self.scenario != TestScenario::Server);
        let runtime = self.runtime.clone();
        let pp = &*self.postprocessor;
        for chunked_samples in samples.chunks(self.batch_size) {
            let input = self.images.prepare_inputs(self.batch_size, chunked_samples);
            if let Some(raw_session) = self.raw_session.as_mut() {
                runtime.block_on(raw_session.run(input)).unwrap();
                let batched_outputs =
                    raw_session.outputs().into_iter().map(|b| b.as_slice()).collect::<Vec<_>>();
                complete_samples(chunked_samples, &batched_outputs, self.batch_size, pp);
            } else {
                self.workload_tx.as_ref().unwrap().send((input, chunked_samples.to_vec())).unwrap();
            }
        }
    }

    fn flush_queries(&mut self) {}

    fn report_latency_results(&mut self, _latencies_ns: &[QuerySampleLatency]) {}
}

impl FuriosaSUT {
    fn new(
        name: CString,
        total_sample_count: usize,
        scenario: TestScenario,
        image_path: Option<PathBuf>,
        mut session_options: SessionOptions,
        run_options: RunOptions,
    ) -> eyre::Result<Self> {
        let runtime = Arc::new(tokio::runtime::Builder::new_current_thread().enable_all().build()?);
        let reader = File::open(META_PATH)?;
        let meta_data: serde_json::Value =
            serde_json::from_reader(reader).expect("JSON is not well-formatted");
        assert!(meta_data.is_object());
        let mut image_map: Vec<usize> = Vec::with_capacity(TOTAL_SAMPLES);
        let images = meta_data.get("images").unwrap().as_array().unwrap();
        for image in images {
            image_map.push(image.get("id").unwrap().as_u64().unwrap().try_into()?);
        }

        let devices = if run_options.optimal_pe_fuse { select_devices(scenario)? } else { vec![] };
        let (preprocessor, binary, main) =
            compile_ssd_small(&devices, run_options.batch_size, scenario)?;

        let postprocessor = build_arc_post_processor(&main, run_options.post_processor);

        if !devices.is_empty() {
            session_options.device(devices.join(","));
        }
        eprintln!("create channel session with options: {:?}", session_options);
        let mut raw_sessions =
            runtime.block_on(session::create_raw_session(&binary, &session_options))?;

        let (raw_session, worker_threads, workload_tx) = if raw_sessions.len() > 1 {
            let (workload_tx, workload_rx) =
                crossbeam::channel::bounded(run_options.worker_num * 5);
            let worker_threads = raw_sessions
                .into_iter()
                .map(|raw_session| {
                    let cloned_workload_rx = workload_rx.clone();
                    let cloned_pp = postprocessor.clone();
                    Some(std::thread::spawn(move || {
                        work_on_thread(
                            raw_session,
                            run_options.batch_size,
                            cloned_workload_rx,
                            cloned_pp,
                        )
                    }))
                })
                .collect::<Vec<_>>();
            (None, worker_threads, Some(workload_tx))
        } else {
            (raw_sessions.pop(), vec![], None)
        };

        Ok(Self {
            runtime,
            name,
            total_sample_count,
            batch_size: run_options.batch_size,
            postprocessor,
            image_map,
            images: ImageStore::new(preprocessor),
            image_path,
            scenario,
            raw_session,
            workload_tx,
            worker_threads,
        })
    }

    fn warmup(&mut self, samples: &[QuerySampleIndex]) {
        // select a random sample from loaded samples.
        let index = samples[samples.len() / 2];
        // QuerySample.id is actually pointer in C++, so 0 can be used as a fake query.
        let samples = (0..(WARMUP_RUNS * self.batch_size))
            .map(|_| QuerySample { id: 0, index })
            .collect::<Vec<_>>();
        run_warmup(self, &samples, self.batch_size, self.scenario);
    }
}

impl Drop for FuriosaSUT {
    fn drop(&mut self) {
        if let Some(raw_session) = self.raw_session.take() {
            drop(raw_session);
        }
        if let Some(workload_tx) = self.workload_tx.take() {
            drop(workload_tx);
        }
        for thread in &mut self.worker_threads {
            if let Some(handle) = thread.take() {
                handle.join().unwrap();
            }
        }
    }
}

pub fn select_devices(scenario: TestScenario) -> eyre::Result<Vec<String>> {
    // Override NPU_DEVNAME environment variable according to the scenario for the optimal performance.
    if scenario == TestScenario::SingleStream {
        return Ok(vec![select_fused_device()?]);
    } else if scenario == TestScenario::Offline {
        return select_non_fused_devices();
    } else if scenario == TestScenario::MultiStream {
        return Ok(vec![select_fused_device()?]);
    }
    eyre::bail!("no optimal device selection supported for the scenario")
}

fn build_arc_post_processor(
    graph: &Graph,
    pp: PostProcessor,
) -> Arc<dyn Postprocess + Sync + Send> {
    match pp {
        PostProcessor::CppPar => Arc::new(CppPostprocessor::new(graph)),
        PostProcessor::RustPar => {
            Arc::new(Postprocessor::new(graph).with_parallel_processing(true))
        }
        PostProcessor::Rust => Arc::new(Postprocessor::new(graph)),
    }
}

fn compile_ssd_small(
    devices: &[String],
    batch_size: usize,
    scenario: TestScenario,
) -> eyre::Result<(GraphExecutor, Vec<u8>, npu_ir::dfg::Graph)> {
    let is_offline_scenario = scenario == TestScenario::Offline;
    if is_offline_scenario {
        unsafe {
            npu_compiler::compile::config::modify_compiler_config(|c| {
                // TODO: Determine if we need to enable this compile option when hw profiling is finished.
                // For now, turning off this option gives better throughput by more than 10%.
                c.graph_lowering_scheme = GraphLoweringScheme::default();
            })
        }
    }
    let (preprocessor, main, binary) = if devices.is_empty() {
        compile("models/ssd_mobilenet_int8.onnx", batch_size, true)?
    } else {
        compile_with_config_from_device("models/ssd_mobilenet_int8.onnx", batch_size, true)?
    };
    Ok((preprocessor, binary, main))
}

fn work_on_thread(
    mut raw_session: RawSession,
    batch_size: usize,
    workload_rx: crossbeam::channel::Receiver<(Vec<Input>, Vec<QuerySample>)>,
    pp: Arc<dyn Postprocess + Sync + Send + 'static>,
) {
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let pp = &*pp;
    while let Ok((inputs, samples)) = workload_rx.recv() {
        if let Err(e) = runtime.block_on(raw_session.run(inputs)) {
            panic!("Failed to inference, {:?}", e);
        }
        let batched_outputs =
            raw_session.outputs().into_iter().map(|b| b.as_slice()).collect::<Vec<_>>();
        complete_samples(&samples, &batched_outputs, batch_size, pp);
    }
}

fn complete_samples(
    samples: &[QuerySample],
    batched_outputs: &[&[u8]],
    batch_size: usize,
    pp: &(dyn Postprocess + Sync + Send + 'static),
) {
    if batch_size == 1 {
        complete_a_sample(&samples[0], batched_outputs, pp);
    } else {
        let chunk_size = batch_size.checked_div(4).unwrap_or(1);
        samples.par_chunks(chunk_size).enumerate().for_each(|(n, query_sample)| {
            for (i, query_sample) in query_sample.iter().enumerate() {
                let n = n * chunk_size + i;
                let outputs = batched_outputs
                    .iter()
                    .map(|b| &b[(n * b.len() / batch_size)..((n + 1) * b.len() / batch_size)])
                    .collect::<Vec<&[u8]>>();
                complete_a_sample(query_sample, &outputs, pp);
            }
        });
    }
}
#[tracing::instrument(
    target = "chrome_layer",
    fields(name = "CompleteASample", cat = "Mlperf"),
    skip(query_sample, data, pp)
)]
fn complete_a_sample(query_sample: &QuerySample, data: &[&[u8]], pp: &dyn Postprocess) {
    let mut result_data = pp.postprocess(query_sample.index as f32, data);
    if query_sample.id != 0 {
        mlperf::query_samples_complete(&mut [result_data.to_query_response(query_sample.id)]);
    }
}

#[derive(Debug, Clone)]
pub struct CppPostprocessor;

impl Postprocess for CppPostprocessor {
    #[allow(clippy::transmute_num_to_bytes)]
    fn postprocess(&self, index: f32, data: &[&[u8]]) -> DetectionResults {
        self.postprocess_impl(index, data)
    }
}

impl CppPostprocessor {
    pub fn new(main: &npu_ir::dfg::Graph) -> Self {
        assert_eq!(main.outputs.len(), NUM_OUTPUTS);

        let mut output_deq_tables = [[0f32; 256]; NUM_OUTPUTS];
        let mut output_exp_scale_deq_tables = [[0f32; 256]; NUM_OUTPUTS];
        let mut score_lowered_shapes: [LoweredShape; NUM_OUTPUTS / 2] =
            [Default::default(); NUM_OUTPUTS / 2];
        let mut box_lowered_shapes: [LoweredShape; NUM_OUTPUTS / 2] =
            [Default::default(); NUM_OUTPUTS / 2];

        for (i, &tensor_index) in main.outputs.iter().enumerate() {
            let tensor = &main[tensor_index];
            let (s, z) = tensor.element_type.get_scale_and_zero_point();
            let mut table = [0f32; 256];
            let mut exp_scale_table = [0f32; 256];
            for q in -128..=127 {
                let index = (q as u8) as usize;
                let x = (s * f64::from(q - z)) as f32;
                if i < 6 {
                    table[index] = f32::exp(x) / (1f32 + f32::exp(x));
                } else {
                    table[index] = x * SCALE_XY;
                    exp_scale_table[index] = f32::exp(x * SCALE_WH);
                };
            }
            if let Some(i) = i.checked_sub(NUM_OUTPUTS / 2) {
                box_lowered_shapes[i] = (&tensor.shape).into();
            } else {
                score_lowered_shapes[i] = (&tensor.shape).into();
            }

            output_deq_tables[i] = table;
            output_exp_scale_deq_tables[i] = exp_scale_table;
        }

        let mut output_base_index = [0usize; 7];
        for i in 0..6 {
            output_base_index[i + 1] = output_base_index[i]
                + NUM_ANCHORS[i] * FEATURE_MAP_SHAPES[i] * FEATURE_MAP_SHAPES[i];
        }

        {
            let score_lowered_shapes_ptr = score_lowered_shapes.as_ptr();
            let box_lowered_shapes_ptr = box_lowered_shapes.as_ptr();
            let output_deq_tables_ptr = output_deq_tables.as_ptr();
            let output_exp_scale_deq_tables_ptr = output_exp_scale_deq_tables.as_ptr();

            let box_priors: Vec<CenteredBox> =
                include_bytes!("../../models/ssd_small_precomputed_priors")
                    .chunks(SIZE_OF_F32 * 4)
                    .map(|bytes| {
                        let (py1, px1, py2, px2) = bytes
                            .chunks(SIZE_OF_F32)
                            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                            .tuples()
                            .next()
                            .unwrap();
                        BoundingBox { py1, px1, py2, px2 }.into()
                    })
                    .collect();
            let box_priors_ptr = box_priors.as_ptr();

            cpp!(unsafe [output_deq_tables_ptr as "float*", output_exp_scale_deq_tables_ptr as "float*", score_lowered_shapes_ptr as "LoweredShapeFromRust*", box_lowered_shapes_ptr as "LoweredShapeFromRust*", box_priors_ptr as "CenteredBox*"] {
                ssd_small::init(output_deq_tables_ptr, output_exp_scale_deq_tables_ptr, score_lowered_shapes_ptr, box_lowered_shapes_ptr, box_priors_ptr);

            });
        }

        Self
    }

    #[allow(clippy::transmute_num_to_bytes)]
    pub fn postprocess_impl(&self, index: f32, data: &[&[u8]]) -> DetectionResults {
        let data_ptr = data.as_ptr();

        let mut ret = Vec::with_capacity(200);
        let result_ptr = ret.as_mut_ptr();

        let n = cpp!(unsafe [index as "float", data_ptr as "const U8Slice*", result_ptr as "DetectionResult*"] -> usize as "size_t" {
            return ssd_small::post_inference<true>(index, data_ptr, result_ptr);
        });
        debug_assert!(n <= 200);
        unsafe {
            ret.set_len(n);
        }

        ret.into()
    }

    #[allow(clippy::transmute_num_to_bytes)]
    pub fn postprocess_impl2(&self, index: f32, data: &TensorArray) -> DetectionResults {
        let data_vec = data.iter_buffers().map(|buffer| buffer.as_slice::<u8>()).collect_vec();
        let data_ptr = data_vec.as_ptr();

        let mut ret = Vec::with_capacity(200);
        let result_ptr = ret.as_mut_ptr();

        let n = cpp!(unsafe [index as "float", data_ptr as "const U8Slice*", result_ptr as "DetectionResult*"] -> usize as "size_t" {
            return ssd_small::post_inference<true>(index, data_ptr, result_ptr);
        });
        debug_assert!(n <= 200);
        unsafe {
            ret.set_len(n);
        }

        ret.into()
    }
}

#[derive(Debug, Clone)]
pub struct Postprocessor {
    output_deq_tables: [[f32; 256]; NUM_OUTPUTS],
    output_exp_scale_deq_tables: [[f32; 256]; NUM_OUTPUTS],
    output_base_index: [usize; 7],
    score_thresholds: [Option<i8>; NUM_OUTPUTS / 2],
    score_lowered_shapes: [LoweredShape; NUM_OUTPUTS / 2],
    box_lowered_shapes: [LoweredShape; NUM_OUTPUTS / 2],
    box_priors: Vec<CenteredBox>,
    parallel_processing: bool,
}

impl Postprocessor {
    pub fn new(main: &npu_ir::dfg::Graph) -> Self {
        assert_eq!(main.outputs.len(), NUM_OUTPUTS);

        let mut output_deq_tables = [[0f32; 256]; NUM_OUTPUTS];
        let mut output_exp_scale_deq_tables = [[0f32; 256]; NUM_OUTPUTS];
        let mut score_lowered_shapes = [Default::default(); NUM_OUTPUTS / 2];
        let mut box_lowered_shapes = [Default::default(); NUM_OUTPUTS / 2];
        let mut score_thresholds = [Default::default(); NUM_OUTPUTS / 2];
        for (i, &tensor_index) in main.outputs.iter().enumerate() {
            let tensor = &main[tensor_index];
            let (s, z) = tensor.element_type.get_scale_and_zero_point();
            let mut table = [0f32; 256];
            let mut exp_scale_table = [0f32; 256];
            for q in -128..=127 {
                let index = (q as u8) as usize;
                let x = (s * f64::from(q - z)) as f32;
                if i < 6 {
                    table[index] = f32::exp(x) / (1f32 + f32::exp(x));
                } else {
                    table[index] = x * SCALE_XY;
                    exp_scale_table[index] = f32::exp(x * SCALE_WH);
                };
            }
            if let Some(i) = i.checked_sub(NUM_OUTPUTS / 2) {
                box_lowered_shapes[i] = (&tensor.shape).into();
            } else {
                score_lowered_shapes[i] = (&tensor.shape).into();

                score_thresholds[i] = (i8::MIN..=i8::MAX).find(|&q| {
                    let index = q as u8 as usize;
                    table[index] > SCORE_THRESHOLD
                });
            }

            output_deq_tables[i] = table;
            output_exp_scale_deq_tables[i] = exp_scale_table;
        }

        let mut output_base_index = [0usize; 7];
        for i in 0..6 {
            output_base_index[i + 1] = output_base_index[i]
                + NUM_ANCHORS[i] * FEATURE_MAP_SHAPES[i] * FEATURE_MAP_SHAPES[i];
        }

        let box_priors = include_bytes!("../../models/ssd_small_precomputed_priors")
            .chunks(SIZE_OF_F32 * 4)
            .map(|bytes| {
                let (py1, px1, py2, px2) = bytes
                    .chunks(SIZE_OF_F32)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .tuples()
                    .next()
                    .unwrap();
                BoundingBox { py1, px1, py2, px2 }.into()
            })
            .collect();

        Self {
            output_deq_tables,
            output_exp_scale_deq_tables,
            output_base_index,
            score_thresholds,
            score_lowered_shapes,
            box_lowered_shapes,
            box_priors,
            parallel_processing: false,
        }
    }

    #[must_use]
    pub fn with_parallel_processing(mut self, x: bool) -> Self {
        self.parallel_processing = x;
        self
    }

    fn filter_result(
        &self,
        query_index: f32,
        scores: &[&[u8]],
        boxes: &[BoundingBox],
        class_index: usize,
        results: &mut Vec<DetectionResult>,
        class_offset: usize,
    ) {
        let mut filtered = Vec::with_capacity(CHANNEL_COUNT);
        for index in 0..6 {
            if let Some(score_threshold) = self.score_thresholds[index] {
                let shape = &self.score_lowered_shapes[index];
                let original_shape =
                    Shape::new(FEATURE_MAP_SHAPES[index], FEATURE_MAP_SHAPES[index]);
                let deq_table = self.output_deq_tables[index];
                for anchor_index in 0..NUM_ANCHORS[index] {
                    for f_y in 0..FEATURE_MAP_SHAPES[index] {
                        for f_x in 0..FEATURE_MAP_SHAPES[index] {
                            let score_index =
                                shape.index(anchor_index * NUM_CLASSES + class_index, f_y, f_x);
                            let q = scores[index][score_index] as i8;
                            if q >= score_threshold {
                                let score = deq_table[q as u8 as usize];
                                debug_assert!(score > SCORE_THRESHOLD);
                                filtered.push((
                                    score,
                                    original_shape.index(anchor_index, f_y, f_x)
                                        + self.output_base_index[index],
                                ));
                            }
                        }
                    }
                }
            }
        }

        filtered.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        for (score, box_index) in filtered {
            let candidate = &boxes[box_index];
            if results[class_offset..].iter().all(|r| candidate.iou(&r.bbox) <= NMS_THRESHOLD) {
                results.push(DetectionResult {
                    index: query_index,
                    bbox: *candidate,
                    score,
                    class: class_index as f32,
                });
            }
        }
    }

    fn filter_results(
        &self,
        query_index: f32,
        scores: &[&[u8]],
        boxes: &[BoundingBox],
    ) -> DetectionResults {
        if self.parallel_processing {
            let mut results = vec![Vec::new(); NUM_CLASSES - 1];
            results.par_iter_mut().enumerate().for_each(|(i, results)| {
                self.filter_result(query_index, scores, boxes, i + 1, results, 0)
            });
            results.into_iter().flatten().collect_vec().into()
        } else {
            let mut results = Vec::with_capacity(CHANNEL_COUNT);
            for class_index in 1..NUM_CLASSES {
                let offset = results.len();
                self.filter_result(query_index, scores, boxes, class_index, &mut results, offset);
            }
            results.into()
        }
    }

    fn decode_box(&self, boxes: &[&[u8]]) -> Vec<BoundingBox> {
        let mut ret = unsafe { uninitialized_vec(CHANNEL_COUNT) };
        let output_deq_tables = &self.output_deq_tables;
        let output_exp_scale_deq_tables = &self.output_exp_scale_deq_tables;
        let output_base_index = &self.output_base_index;
        let box_lowered_shapes = &self.box_lowered_shapes;

        for index in 0..boxes.len() {
            let anchor_stride = ANCHOR_STRIDES[index];
            let deq_table = &output_deq_tables[index + 6];
            let exp_scale_table = &output_exp_scale_deq_tables[index + 6];
            let shape = &box_lowered_shapes[index];
            let b = boxes[index];
            debug_assert!(!b.is_empty());
            for anchor_index in 0..NUM_ANCHORS[index] {
                for f_y in 0..FEATURE_MAP_SHAPES[index] {
                    for f_x in 0..FEATURE_MAP_SHAPES[index] {
                        let feature_index = f_y * FEATURE_MAP_SHAPES[index] + f_x;
                        // TODO `b` 를 cache-friendly 하게 접근하는 lowered shape 고민
                        let q0 = b[shape.index(anchor_index * 4, f_y, f_x)];
                        let q1 = b[shape.index(anchor_index * 4 + 1, f_y, f_x)];
                        let q2 = b[shape.index(anchor_index * 4 + 2, f_y, f_x)];
                        let q3 = b[shape.index(anchor_index * 4 + 3, f_y, f_x)];

                        let bx = CenteredBox {
                            pcy: deq_table[q0 as usize],
                            pcx: deq_table[q1 as usize],
                            ph: exp_scale_table[q2 as usize],
                            pw: exp_scale_table[q3 as usize],
                        };

                        let box_index =
                            output_base_index[index] + feature_index + anchor_index * anchor_stride;
                        // TODO `prior_index` 대신 `box_index` 를 사용할 수 있도록 `self.box_priors` 의 배열 변경
                        let prior_index = output_base_index[index]
                            + feature_index * NUM_ANCHORS[index]
                            + anchor_index;

                        ret[box_index] = self.box_priors[prior_index].adjust(bx).into();
                    }
                }
            }
        }
        ret
    }

    pub fn postprocess_impl(&self, index: f32, data: &[&[u8]]) -> DetectionResults {
        let (scores, boxes) = data.split_at(6);
        let boxes = self.decode_box(boxes);
        debug_assert_eq!(boxes.len(), CHANNEL_COUNT);
        self.filter_results(index, scores, &boxes)
    }
}

impl Postprocess for Postprocessor {
    #[tracing::instrument(
        target = "chrome_layer",
        fields(name = "PostProcess", cat = "Mlperf"),
        skip(self, data)
    )]
    fn postprocess(&self, index: f32, data: &[&[u8]]) -> DetectionResults {
        self.postprocess_impl(index, data)
    }
}

/// The interface a client implements to coordinate with the loadgen
/// which samples should be loaded
impl QuerySampleLibrary for FuriosaSUT {
    /// A human readable name for the model
    fn name(&self) -> &std::ffi::CStr {
        self.name.as_c_str()
    }

    /// Total number of samples in library
    fn total_sample_count(&self) -> usize {
        TOTAL_SAMPLES
    }

    /// The number of samples that are guaranteed to fit in RAM
    fn performance_sample_count(&self) -> usize {
        // This value is overridden by the value in TestSettings.performance_sample_count_override if non-zero.
        // https://github.com/mlcommons/inference/blob/56c0d7a816c7f67077fa7651a3584cac116e0633/loadgen/test_settings_internal.cc#L115
        // Value brought from https://github.com/mlcommons/inference/blob/master/mlperf.conf
        256
    }

    /// Loads the requested query samples into memory
    /// Paired with calls to UnloadSamplesFromRam
    /// A previously loaded sample will not be loaded again
    #[tracing::instrument(
        target = "chrome_layer",
        fields(name = "LoadSamplesToRam", cat = "Mlperf"),
        skip(self, samples)
    )]
    fn load_samples_to_ram(&mut self, samples: &[QuerySampleIndex]) {
        for &index in samples {
            // In test environment, we may have only small number of samples with low index numbers.
            // To load samples up to performance sample count and to measure the memory and cache performance,
            // reload same files to another buffer even if self.total_sample_count < TOTAL_SAMPLES.
            // Accuracy test will fail in that case.
            let wrapped_index = index % self.total_sample_count;
            let data_path =
                if let Some(path) = &self.image_path { path.to_str().unwrap() } else { DATA_PATH };
            let filename = format!("{}{:012}.png.raw", data_path, self.image_map[wrapped_index]);
            self.images.load_image(index, filename, self.batch_size);
        }
        self.warmup(samples);
    }

    /// Unloads the requested query samples from memory
    /// A previously unloaded sample will not be unloaded
    #[tracing::instrument(
        target = "chrome_layer",
        fields(name = "UnloadSamplesFromRam", cat = "Mlperf"),
        skip(self, samples)
    )]
    fn unload_samples_from_ram(&mut self, samples: &[QuerySampleIndex]) {
        for &index in samples {
            self.images.unload_image(index);
        }
    }
}

pub fn run(
    image_path: Option<PathBuf>,
    scenario: TestScenario,
    mode: TestMode,
    config_path: impl AsRef<Path>,
    session_options: SessionOptions,
    run_options: RunOptions,
) {
    if scenario == TestScenario::SingleStream {
        run_with_blocking_session(
            image_path,
            scenario,
            mode,
            config_path,
            session_options,
            run_options,
        );
    } else {
        run_with_tokio(image_path, scenario, mode, config_path, session_options, run_options)
    }
}

pub fn run_with_tokio(
    image_path: Option<PathBuf>,
    scenario: TestScenario,
    mode: TestMode,
    config_path: impl AsRef<Path>,
    session_options: SessionOptions,
    run_options: RunOptions,
) {
    let total_sample_count = if image_path.is_none() { 10 } else { TOTAL_SAMPLES };
    let model = CString::new("ssd-mobilenet").unwrap();
    let test_settings = prepare_test_settings(config_path, &model, scenario, mode);
    let log_settings = prepare_log_settings("mlperf_ssd_small_out");
    let enable_tokio_console = run_options.enable_tokio_console;

    let mut sut = FuriosaSUT::new(
        model,
        total_sample_count,
        scenario,
        image_path,
        session_options,
        run_options,
    )
    .unwrap();

    eprintln!("start test");
    if enable_tokio_console {
        console_subscriber::init();
    }
    mlperf::start_test_ex(&mut sut, &test_settings, &log_settings);
    drop(sut);
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader};

    use super::*;
    use crate::utils::{prepare_raw_output, to_graph};

    #[test]
    fn unittest_ssd_small_postprocess() -> eyre::Result<()> {
        let model_path = "models/ssd_mobilenet_int8.onnx";
        let graph: npu_ir::dfg::Graph = to_graph(model_path, 1, true)?.try_into()?;
        let pp = CppPostprocessor::new(&graph);

        let output =
            prepare_raw_output("coco-300-golden/raw/000000331352.png.raw", graph, model_path)
                .unwrap();

        let data = output.iter().map(|b| b.as_slice::<u8>()).collect::<Vec<_>>();
        assert_eq!(
            pp.postprocess(0., &data).as_f32_slice(),
            [
                0.0,
                0.5035858,
                0.1345388,
                0.996843,
                0.83921313,
                0.960106,
                70.0,
                0.0,
                0.10253234,
                0.14891449,
                0.4925146,
                0.85108554,
                0.67053676,
                81.0,
                0.0,
                0.110852465,
                0.17115179,
                0.33082718,
                0.8310759,
                0.31760606,
                81.0
            ]
        );
        Ok(())
    }

    #[test]
    fn unittest_mlperf_ssd_small() {
        let config_path = "models/mlperf_small.conf";
        run(
            None,
            TestScenario::SingleStream,
            TestMode::PerformanceOnly,
            config_path,
            SessionOptions::default(),
            RunOptions {
                optimal_pe_fuse: true,
                batch_size: 1,
                enable_tokio_console: false,
                ..Default::default()
            },
        );
        let file = File::open("mlperf_ssd_small_out/mlperf_log_summary.txt").unwrap();
        let reader = BufReader::new(file);
        assert!(reader.lines().any(|line| line.unwrap().contains("Result is : VALID")));
    }

    #[test]
    fn profile_ssd_small_post_processing() -> eyre::Result<()> {
        let model_path = "models/ssd_mobilenet_int8.onnx";
        let graph: npu_ir::dfg::Graph = to_graph(model_path, 1, true)?.try_into()?;
        let pp = CppPostprocessor::new(&graph);

        let output =
            prepare_raw_output("coco-300-golden/raw/000000331352.png.raw", graph, model_path)
                .unwrap();
        let data = output.iter().map(|b| b.as_slice::<u8>()).collect::<Vec<_>>();

        for i in 0..100 {
            pp.postprocess(i as f32, &data);
        }

        let guard = pprof::ProfilerGuard::new(1000)?;
        for i in 0..10000 {
            pp.postprocess(i as f32, &data);
        }
        if let Ok(report) = guard.report().build() {
            let file = File::create("target/ssd_small.svg")?;
            report.flamegraph(file)?;
        };
        Ok(())
    }
}
