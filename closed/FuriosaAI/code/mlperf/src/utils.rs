use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::sync::{Condvar, Mutex};
use std::{env, fs, mem, slice, thread};

use cached_persistence::proc_macro::cached_persistence;
use nux::telemetry::{ProfilerWriter, TelemetryGuard};
use onnx::ModelProto;
use prost::Message;
use structopt::StructOpt;
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

use loadgen::mlperf::{
    LogSettings, QuerySample, QuerySampleResponse, ResponseId, SystemUnderTest, TestMode,
    TestScenario, TestSettings,
};
use npu_compiler::compile::gir_to_lir::ga::{GaParameter, GenomeBuilderTactic};
use npu_compiler::compile::{DfgToLdfg, LdfgToEnf, OnnxToDfg};
use npu_executor::traits::Execute;
use npu_executor::GraphExecutor;
use npu_executor::IntoExecutor;
use npu_ir::furiosa_ir::{FuriosaIr, FuriosaIrOnnx};
use npu_ir::traits::Compile;
use npu_ir::{AxisIndex, Buffer, LabeledShape, LoweredActivationShape, TensorShape};
use nux::async_nux::session::Input;
use nux::RequestId;

#[derive(Clone)]
struct ImageEntry {
    // store `ptr` & `len` to avoid referencing `Arc`
    ptr: *mut u8,
    len: usize,
    _buffer: Buffer,
}

pub struct ImageStore {
    images: Vec<Option<ImageEntry>>,
    preprocessor: GraphExecutor,
    dummy_buffer: Vec<u8>,
}

impl ImageStore {
    pub fn new(preprocessor: GraphExecutor) -> Self {
        Self { images: Default::default(), preprocessor, dummy_buffer: Default::default() }
    }

    pub fn load_image(&mut self, index: usize, file: impl AsRef<Path>, batch_size: usize) {
        // This is not an optimized preprocessing implementation.
        // Assigns buffer for the batch size, but uses only the first chunk of the buffer for preprocessing.
        let mut buffer = fs::read(file).unwrap();
        let input_size = buffer.len();
        buffer.resize(input_size * batch_size, 0);

        let result_buffer = self.preprocessor.execute(&[buffer.into()]).unwrap();
        if self.images.len() <= index {
            self.images.resize(index + 1, None);
        }

        let buffer = Buffer::from_slice(
            &result_buffer[0].as_slice::<u8>()[..(result_buffer[0].len() / batch_size)],
        );
        let ptr = buffer.as_ptr::<u8>() as *mut u8;
        let len = buffer.len();
        self.images[index] = Some(ImageEntry { ptr, len, _buffer: buffer });

        // If the dummy buffer is empty, set one
        if self.dummy_buffer.is_empty() {
            self.dummy_buffer = vec![0u8; len * batch_size];
        }
    }

    pub fn unload_image(&mut self, index: usize) {
        self.images[index] = None;
    }

    pub fn buffer(&self, index: usize) -> Buffer {
        let ImageEntry { ptr, len, _buffer } = &self.images[index].as_ref().unwrap();
        unsafe { Buffer::from_raw_parts(*ptr, *len, dummy_free) }
    }

    pub fn slice(&self, index: usize) -> &[u8] {
        let ImageEntry { ptr, len, _buffer } = &self.images[index].as_ref().unwrap();
        unsafe { slice::from_raw_parts(*ptr, *len) }
    }

    pub fn prepare_inputs(&self, batch_size: usize, samples: &[QuerySample]) -> Vec<Input> {
        let mut scattered_buffer = Vec::with_capacity(batch_size);
        for sample in samples {
            scattered_buffer.push(self.buffer(sample.index));
        }
        for _ in scattered_buffer.len()..batch_size {
            scattered_buffer.push(unsafe {
                Buffer::from_raw_parts(
                    self.dummy_buffer.as_ptr() as *mut u8,
                    scattered_buffer[0].len(),
                    dummy_free,
                )
            })
        }
        vec![Input::Scattered(scattered_buffer)]
    }
}

extern "C" fn dummy_free(_: *mut u8, _: usize) {}

pub struct RunOptions {
    pub optimal_pe_fuse: bool,
    pub batch_size: usize,
    pub enable_tokio_console: bool,
    pub post_processor: PostProcessor,
    pub worker_num: usize,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            optimal_pe_fuse: false,
            batch_size: 1,
            enable_tokio_console: true,
            post_processor: PostProcessor::default(),
            worker_num: 1,
        }
    }
}

impl RunOptions {
    pub fn optimal_pe_fuse(&mut self, optimal_pe_fuse: bool) -> &mut Self {
        self.optimal_pe_fuse = optimal_pe_fuse;
        self
    }

    pub fn batch_size(&mut self, batch_size: usize) -> &mut Self {
        if batch_size == 0 {
            eprintln!("batch size should be larger than 0, it will be set as 1");
            self.batch_size = 1;
        } else {
            self.batch_size = batch_size;
        }
        self
    }

    pub fn worker_num(&mut self, worker_num: usize) -> &mut Self {
        if worker_num == 0 {
            eprintln!("worker num should be larger than 0, it will be set as 1");
            self.worker_num = 1;
        } else {
            self.worker_num = worker_num;
        }
        self
    }
}

pub trait Postprocess {
    fn postprocess(&self, index: f32, data: &[&[u8]]) -> DetectionResults;
}

// 0~5 scores 6~11 boxes
const NUM_OUTPUTS: usize = 12;

#[derive(Default)]
pub struct OutputTensorMap {
    map: HashMap<RequestId, OutputTensors>,
}

impl OutputTensorMap {
    pub fn push(&mut self, request_id: RequestId, index: usize, buffer: Vec<u8>) {
        let output_tensors = self.map.entry(request_id).or_insert_with(OutputTensors::new);
        output_tensors.push(index, buffer)
    }

    pub fn is_full(&self, request_id: RequestId) -> bool {
        if let Some(output_tensors) = self.map.get(&request_id) {
            output_tensors.is_full()
        } else {
            false
        }
    }

    pub fn reset(&mut self, request_id: RequestId) {
        let output_tensors = self.map.entry(request_id).or_insert_with(OutputTensors::new);
        *output_tensors = OutputTensors::new()
    }

    pub fn get(&self, request_id: &RequestId) -> Option<&OutputTensors> {
        self.map.get(request_id)
    }
}

#[derive(Default)]
pub struct OutputTensors {
    pub counter: usize,
    pub buffer: Vec<Vec<u8>>,
}

impl OutputTensors {
    pub fn new() -> Self {
        Self { counter: 0, buffer: vec![vec![]; NUM_OUTPUTS] }
    }

    pub fn push(&mut self, index: usize, buffer: Vec<u8>) {
        self.counter += 1;
        assert!(self.buffer[index].is_empty());
        self.buffer[index] = buffer;
    }

    pub fn is_full(&self) -> bool {
        self.counter == NUM_OUTPUTS
    }
}

pub fn iou(b1: &[f32], b2: &[f32]) -> f32 {
    let clamp_lower = |x: f32| {
        if x < 0f32 {
            0f32
        } else {
            x
        }
    };
    let area1 = clamp_lower(b1[3] - b1[1]) * clamp_lower(b1[2] - b1[0]);
    let area2 = clamp_lower(b2[3] - b2[1]) * clamp_lower(b2[2] - b2[0]);
    let cw = clamp_lower(f32::min(b1[3], b2[3]) - f32::max(b1[1], b2[1]));
    let ch = clamp_lower(f32::min(b1[2], b2[2]) - f32::max(b1[0], b2[0]));
    let overlap = cw * ch;
    overlap / (area1 + area2 - overlap + 0.00001f32)
}

pub struct Shape {
    height: usize,
    width: usize,
}

impl Shape {
    pub fn new(h: usize, w: usize) -> Self {
        Self { height: h, width: w }
    }

    pub fn index(&self, c: usize, h: usize, w: usize) -> usize {
        c * self.height * self.width + h * self.width + w
    }
}

pub struct WorkloadControl {
    val: Mutex<(usize, usize)>, // (available_load, remaining_warmups)
    cond: Condvar,
}

impl WorkloadControl {
    pub fn new(max_load: usize) -> Self {
        Self { val: Mutex::new((max_load, 0)), cond: Condvar::new() }
    }

    pub fn acquire(&self, cnt: usize) {
        let mut val = self.val.lock().unwrap();
        while (*val).0 < cnt {
            val = self.cond.wait(val).unwrap();
        }
        (*val).0 -= cnt;
    }

    pub fn release(&self, cnt: usize) {
        let mut val = self.val.lock().unwrap();
        (*val).0 += cnt;
        let remaining_warmups = (*val).1;
        if remaining_warmups > cnt {
            (*val).1 = remaining_warmups - cnt;
        } else if remaining_warmups > 0 {
            (*val).1 = 0;
        }
        self.cond.notify_all();
    }

    pub fn set_warmup_runs(&self, warmups: usize) {
        let mut val = self.val.lock().unwrap();
        (*val).1 = warmups;
    }

    pub fn wait_warmup_finished(&self) {
        let mut val = self.val.lock().unwrap();
        while (*val).1 > 0 {
            val = self.cond.wait(val).unwrap();
        }
    }
}

pub fn run_warmup<T: SystemUnderTest>(
    sut: &mut T,
    samples: &[QuerySample],
    batch_size: usize,
    scenario: TestScenario,
) {
    // To ensure first inference has finished.
    let first_query_size = if batch_size < samples.len() { batch_size } else { samples.len() };
    sut.issue_query(&samples[..first_query_size]);
    eprintln!("finished first warmup inference: {} batch", first_query_size);

    let query_size = if scenario == TestScenario::SingleStream {
        batch_size // maybe 1
    } else {
        samples.len() - first_query_size // all remaining samples in one query
    };
    for q in samples[first_query_size..].chunks(query_size) {
        sut.issue_query(q);
    }
}

pub fn init_nux_profiler() -> Option<TelemetryGuard> {
    std::env::var("NUX_PROFILER_PATH").ok().map(|path| {
        nux::telemetry::new()
            .with_logger(None)
            .with_profiler(ProfilerWriter::File(path))
            .init()
            .unwrap()
    })
}

pub fn init_logger() {
    let env_filter = EnvFilter::from_default_env();
    let fmt_layer = tracing_subscriber::fmt::layer();
    let _ = tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer.with_filter(filter_fn(|metadata| metadata.target() != "chrome_layer")))
        .try_init();
}

pub fn prepare_test_settings(
    config_path: impl AsRef<Path>,
    model: impl AsRef<CStr>,
    scenario: TestScenario,
    mode: TestMode,
) -> TestSettings {
    let mut test_settings = TestSettings::default();
    let scenario_name = match scenario {
        TestScenario::SingleStream => CString::new("SingleStream").unwrap(),
        TestScenario::MultiStream => CString::new("MultiStream").unwrap(),
        TestScenario::Server => CString::new("Server").unwrap(),
        TestScenario::Offline => CString::new("Offline").unwrap(),
    };
    test_settings.from_config(config_path, model, scenario_name).unwrap();
    test_settings.scenario = scenario;
    test_settings.mode = mode;
    test_settings
}

pub fn prepare_log_settings(log_path: &str) -> LogSettings {
    let mut log_settings = LogSettings::default();
    std::fs::create_dir_all(log_path).unwrap();
    log_settings.log_output.outdir = CString::new(log_path).unwrap();
    log_settings
}

// TODO: replace this function when a new device API is released.
// Assumes that NPU_NPUNAME or NPU_DEVNAME environment variable is set.
fn npu_name() -> Option<String> {
    if let Ok(npu_name) = env::var("NPU_NPUNAME") {
        return Some(npu_name);
    } else if let Ok(dev_name) = env::var(npu_config::config::NPU_DEVNAME_ENV) {
        return if let Some(pe_pos) = dev_name.find("pe") {
            Some(String::from(&dev_name[..pe_pos]))
        } else {
            Some(dev_name)
        };
    }
    None
}

// TODO: replace this function when a new device API is released.
pub fn select_non_fused_devices() -> eyre::Result<Vec<String>> {
    if let Some(npu_name) = npu_name() {
        let pe0 = format!("{}pe0", npu_name);
        let pe1 = format!("{}pe1", npu_name);
        // NPU_DEVNAME environment variable does not support comma separated device names.
        // Set only one of the selected devices.
        env::set_var(npu_config::config::NPU_DEVNAME_ENV, &pe0);
        // TODO: support next version NPUs
        env::set_var("NPU_GLOBAL_CONFIG_PATH", npu_config::npuid::WARBOY);
        return Ok(vec![pe0, pe1]);
    }
    eyre::bail!("cannot determine NPU_DEVNAME")
}

// TODO: replace this function when a new device API is released.
pub fn select_fused_device() -> eyre::Result<String> {
    if let Some(npu_name) = npu_name() {
        let fused = format!("{}pe0-1", npu_name);
        env::set_var(npu_config::config::NPU_DEVNAME_ENV, &fused);
        // TODO: support next version NPUs
        env::set_var("NPU_GLOBAL_CONFIG_PATH", npu_config::npuid::WARBOY_2PE);
        return Ok(fused);
    }
    eyre::bail!("cannot determine NPU_DEVNAME")
}

fn hidden_compiler_key() -> (String, String) {
    // It's possible to use npu_config::config::Config instead of String if we add derive(Eq, Hash) for the type.
    let npu_config = serde_json::to_string(npu_config::config::global_config()).unwrap();

    // It's tricky to make npu_compiler::compile::compile::Config implement Eq or Hash since it contains f64 fields.
    // So, just use serialized value for the cache key.
    let compiler_config = serde_json::to_string(npu_compiler::compile::compiler_config()).unwrap();

    (npu_config, compiler_config)
}

fn to_graph_key(
    model_path: impl AsRef<Path>,
    batch_size: usize,
    remove_unlower: bool,
) -> (String, usize, bool, String, String) {
    // If the content of the model_path file itself is used for cache key,
    // the cached file size and read/decoding time increases two times.
    // We use checksum of the model_path file instead.
    let file_checksum = checksums::hash_file(model_path.as_ref(), checksums::Algorithm::SHA2512);

    // These configurations affect the output of compiler.
    let (npu_config, compiler_config) = hidden_compiler_key();

    (file_checksum, batch_size, remove_unlower, npu_config, compiler_config)
}

#[cached_persistence(path = "cached/ldfg")]
#[cached(
    key = "(String, usize, bool, String, String)",
    convert = r#"{ to_graph_key(&model_path, batch_size, remove_unlower) }"#
)]
fn to_graph_cached(
    model_path: impl AsRef<Path>,
    batch_size: usize,
    remove_unlower: bool,
) -> Result<npu_compiler::dfg::Graph, String> {
    // eyre::Report does not implement Serialize and DeserializeOwned
    eprintln!("no cached result for to_graph: compiling and generating cache");
    to_graph_inner(model_path, batch_size, remove_unlower).map_err(|e| {
        format!(
            "
  this might be cached error result from previous run
  consider deleting cache or setting `MLCOMMONS_USE_CACHED_GRAPH=false`
  if this error is unexpected and hard to debug

{:?}",
            e
        )
    })
}

fn update_batch(graph: &mut npu_ir::dfg::Graph, n: usize) {
    for tensor_index in graph.activations() {
        let batch_size = &mut graph[tensor_index].shape[AxisIndex::from(0)];
        assert_eq!(*batch_size, 1);
        *batch_size = n;
    }
}

fn to_graph_inner(
    model_path: impl AsRef<Path>,
    batch_size: usize,
    remove_unlower: bool,
) -> eyre::Result<npu_compiler::dfg::Graph> {
    let buffer = std::fs::read(model_path)?;
    let model = ModelProto::decode(buffer.as_slice())?;
    let furiosa_ir_onnx = FuriosaIrOnnx::new(model)?;
    let mut dfg = OnnxToDfg.compile(&furiosa_ir_onnx)?;
    update_batch(&mut dfg, batch_size);
    let mut dfg_to_ldfg = DfgToLdfg::default();
    dfg_to_ldfg.remove_unlower = remove_unlower;
    let ldfg = dfg_to_ldfg.compile(dfg)?;
    npu_compiler::dfg::Graph::try_from(ldfg)
}

pub fn to_graph(
    model_path: impl AsRef<Path>,
    batch_size: usize,
    remove_unlower: bool,
) -> eyre::Result<npu_compiler::dfg::Graph> {
    // Common compiler config should be set here.
    unsafe {
        npu_compiler::compile::config::modify_compiler_config(|config| {
            config.optimize_by_genetic_algorithm = Some(GaParameter {
                population_size: 400,
                generation_limit: 1000,
                max_prefetch_size: 30,
                genome_builder_tactic: GenomeBuilderTactic::RandomGenomeBuilder,
                ..Default::default()
            });
            config.ignore_default_pdb = false;
        });
    }

    if env::var("MLCOMMONS_USE_CACHED_GRAPH").into_iter().any(|v| v == "true") {
        to_graph_cached(model_path, batch_size, remove_unlower).map_err(|e| eyre::eyre!(e))
    } else {
        to_graph_inner(model_path, batch_size, remove_unlower)
    }
}

fn to_enf_binary_key(ldfg: &npu_ir::ldfg::LoweredGraph) -> (String, String, String) {
    // LoweredGraph type does not implement Hash and Eq traits, so manually serialized here.
    // Use checksum for performance and disk usage.
    let ldfg = ldfg.to_buffer().unwrap();
    let mut ldfg = ldfg.as_slice();
    let graph_checksum = checksums::hash_reader(&mut ldfg, checksums::Algorithm::SHA2512);

    // These configurations affect the output of compiler.
    let (npu_config, compiler_config) = hidden_compiler_key();

    (graph_checksum, npu_config, compiler_config)
}

#[cached_persistence(path = "cached/enf")]
#[cached(key = "(String, String, String)", convert = r#"{ to_enf_binary_key(&ldfg) }"#)]
fn to_enf_binary_cached(ldfg: npu_ir::ldfg::LoweredGraph) -> Result<Vec<u8>, String> {
    // eyre::Report does not implement Serialize and DeserializeOwned
    eprintln!("no cached result for to_enf_binary: compiling and generating cache");
    to_enf_binary_inner(ldfg).map_err(|e| {
        format!(
            "
  this might be cached error result from previous run
  consider deleting cache or setting `MLCOMMONS_USE_CACHED_GRAPH=false`
  if this error is unexpected and hard to debug

{:?}",
            e
        )
    })
}

fn to_enf_binary_inner(ldfg: npu_ir::ldfg::LoweredGraph) -> eyre::Result<Vec<u8>> {
    let executable = LdfgToEnf::default().compile(ldfg)?;
    FuriosaIr::Executable(executable.into()).serialize()
}

fn to_enf_binary(ldfg: npu_ir::ldfg::LoweredGraph) -> eyre::Result<Vec<u8>> {
    if env::var("MLCOMMONS_USE_CACHED_GRAPH").into_iter().any(|v| v == "true") {
        to_enf_binary_cached(ldfg).map_err(|e| eyre::eyre!(e))
    } else {
        to_enf_binary_inner(ldfg)
    }
}

pub fn compile(
    model_path: impl AsRef<Path>,
    batch_size: usize,
    remove_unlower: bool,
) -> eyre::Result<(GraphExecutor, npu_ir::dfg::Graph, Vec<u8>)> {
    eprintln!("compile: {}, {}, {}", model_path.as_ref().display(), batch_size, remove_unlower);

    let mut graph = to_graph(model_path, batch_size, remove_unlower)?;

    let preprocessing = graph.extract_preprocessing_graph()?;
    for &tensor_index in preprocessing.outputs() {
        assert!(
            graph[tensor_index].shape.is_lowered_activation(),
            "preprocessing should involve tensor-lowering"
        );
    }
    let preprocessor = preprocessing.into_dfg_of_single_subgraph()?.into_executor()?;

    let main = graph.into_dfg_of_single_subgraph()?;

    let binary = to_enf_binary(main.clone().try_into()?)?;

    Ok((preprocessor, main, binary))
}

pub fn compile_with_config_from_device(
    model_path: impl AsRef<Path>,
    batch_size: usize,
    remove_unlower: bool,
) -> eyre::Result<(GraphExecutor, npu_ir::dfg::Graph, Vec<u8>)> {
    let model_path = model_path.as_ref().to_owned();
    // use a thread to confine the effect of modifying the global config
    thread::spawn(move || {
        unsafe {
            let npu_spec = npu_config::config::load_from_library()?;
            npu_config::config::modify_global_config(|c| {
                *c = npu_spec;
            });
        }
        compile(model_path, batch_size, remove_unlower)
    })
    .join()
    .unwrap()
}

fn collect_output_shapes(graph: &npu_ir::dfg::Graph) -> Vec<TensorShape> {
    graph.outputs.iter().map(|&tensor_index| graph[tensor_index].shape.clone()).collect()
}

#[cached_persistence(path = "cached/outputs")]
#[cached(
    key = "(PathBuf, PathBuf, Vec<TensorShape>)",
    convert = r#"{ (input_path.as_ref().to_owned(), _model_path.as_ref().to_owned(), collect_output_shapes(&graph)) }"#
)]
pub fn prepare_raw_output(
    input_path: impl AsRef<Path>,
    graph: npu_ir::dfg::Graph,
    _model_path: impl AsRef<Path>,
) -> Option<Vec<Buffer>> {
    eprintln!("prepare_raw_output: {}", input_path.as_ref().display());
    eprintln!("output shapes: {:?}", collect_output_shapes(&graph));

    let input_buffer = fs::read(input_path).ok()?.into();
    let mut executor = graph.into_executor().ok()?;
    executor.execute(&[input_buffer]).ok()
}

#[derive(Debug, Default, Clone, Copy)]
pub struct LoweredShape {
    ho_stride: usize,
    co_stride: usize,
    hi_stride: usize,
    ci_stride: usize,
    w_stride: usize,
    slice_height: usize,
    slice_channel: usize,
}

impl<'a> From<&'a TensorShape> for LoweredShape {
    fn from(shape: &'a TensorShape) -> Self {
        match shape {
            TensorShape::LabeledShape { inner } => inner.into(),
            TensorShape::LoweredActivationShape { inner } => inner.into(),
            _ => unimplemented!("Unsupported lowered shape: {}", shape),
        }
    }
}

impl<'a> From<&'a LabeledShape> for LoweredShape {
    fn from(shape: &'a LabeledShape) -> Self {
        assert!(shape.is_nchw());
        Self {
            ho_stride: 0,
            co_stride: 0,
            hi_stride: shape.width().unwrap(),
            ci_stride: shape.height().unwrap() * shape.width().unwrap(),
            w_stride: 1,
            slice_height: shape.height().unwrap(),
            slice_channel: shape.channel().unwrap(),
        }
    }
}

impl<'a> From<&'a LoweredActivationShape> for LoweredShape {
    fn from(shape: &'a LoweredActivationShape) -> Self {
        if shape.is_nhoco_hcw() {
            Self {
                ho_stride: shape.inner_partitions() * shape.slice_volume(),
                co_stride: shape.slice_volume(),
                hi_stride: shape.slice_channel() * shape.slice_width(),
                ci_stride: shape.slice_width(),
                w_stride: 1,
                slice_height: shape.slice_height(),
                slice_channel: shape.slice_channel(),
            }
        } else if shape.is_nhoco_hwc() {
            Self {
                ho_stride: shape.inner_partitions() * shape.slice_volume(),
                co_stride: shape.slice_volume(),
                hi_stride: shape.slice_channel() * shape.slice_width(),
                w_stride: shape.slice_channel(),
                ci_stride: 1,
                slice_height: shape.slice_height(),
                slice_channel: shape.unaligned_slice_channel(),
            }
        } else {
            unimplemented!("Unsupported lowered shape: {}", shape);
        }
    }
}

impl LoweredShape {
    pub fn index(&self, c: usize, h: usize, w: usize) -> usize {
        let ho = h / self.slice_height;
        let hi = h % self.slice_height;
        let co = c / self.slice_channel;
        let ci = c % self.slice_channel;
        ho * self.ho_stride
            + co * self.co_stride
            + hi * self.hi_stride
            + ci * self.ci_stride
            + w * self.w_stride
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct BoundingBox {
    pub py1: f32,
    pub px1: f32,
    pub py2: f32,
    pub px2: f32,
}

impl From<CenteredBox> for BoundingBox {
    fn from(cbox: CenteredBox) -> Self {
        Self { py1: cbox.py1(), px1: cbox.px1(), py2: cbox.py2(), px2: cbox.px2() }
    }
}

impl BoundingBox {
    #[no_mangle]
    pub extern "C" fn new_bounding_box(py1: f32, px1: f32, py2: f32, px2: f32) -> BoundingBox {
        BoundingBox { py1, px1, py2, px2 }
    }

    #[inline]
    pub fn pw(&self) -> f32 {
        self.px2 - self.px1
    }

    #[inline]
    pub fn ph(&self) -> f32 {
        self.py2 - self.py1
    }

    #[inline]
    pub fn pcx(&self) -> f32 {
        self.px1 + self.pw() * 0.5
    }

    #[inline]
    pub fn pcy(&self) -> f32 {
        self.py1 + self.ph() * 0.5
    }

    #[inline]
    pub fn area(&self) -> f32 {
        self.pw() * self.ph()
    }

    #[inline]
    pub fn transpose(&mut self) {
        mem::swap(&mut self.px1, &mut self.py1);
        mem::swap(&mut self.px2, &mut self.py2);
    }

    #[inline]
    #[must_use]
    pub fn into_transposed(mut self) -> Self {
        self.transpose();
        self
    }

    #[inline]
    pub fn iou(&self, other: &Self) -> f32 {
        let clamp_lower = |x: f32| {
            if x < 0f32 {
                0f32
            } else {
                x
            }
        };
        let cw = clamp_lower(f32::min(self.px2, other.px2) - f32::max(self.px1, other.px1));
        let ch = clamp_lower(f32::min(self.py2, other.py2) - f32::max(self.py1, other.py1));
        let overlap = cw * ch;
        overlap / (self.area() + other.area() - overlap + 0.00001f32)
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct CenteredBox {
    pub pcy: f32,
    pub pcx: f32,
    pub ph: f32,
    pub pw: f32,
}

impl From<BoundingBox> for CenteredBox {
    fn from(prior: BoundingBox) -> Self {
        Self { pw: prior.pw(), ph: prior.ph(), pcx: prior.pcx(), pcy: prior.pcy() }
    }
}

impl CenteredBox {
    #[no_mangle]
    pub extern "C" fn new_centered_box(pcy: f32, pcx: f32, ph: f32, pw: f32) -> CenteredBox {
        CenteredBox { pcy, pcx, ph, pw }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        vec![self.pcy, self.pcx, self.ph, self.pw]
    }

    #[inline]
    #[no_mangle]
    pub extern "C" fn px1(&self) -> f32 {
        self.pcx - self.pw * 0.5
    }

    #[inline]
    #[no_mangle]
    pub extern "C" fn px2(&self) -> f32 {
        self.pcx + self.pw * 0.5
    }

    #[inline]
    #[no_mangle]
    pub extern "C" fn py1(&self) -> f32 {
        self.pcy - self.ph * 0.5
    }

    #[inline]
    #[no_mangle]
    pub extern "C" fn py2(&self) -> f32 {
        self.pcy + self.ph * 0.5
    }

    #[inline]
    pub fn transpose(&mut self) {
        mem::swap(&mut self.pcx, &mut self.pcy);
        mem::swap(&mut self.pw, &mut self.ph);
    }

    #[inline]
    #[must_use]
    pub fn into_transposed(mut self) -> Self {
        self.transpose();
        self
    }

    #[must_use]
    #[inline]
    #[no_mangle]
    pub extern "C" fn adjust(&self, x: Self) -> Self {
        Self {
            pcx: self.pcx + x.pcx * self.pw,
            pcy: self.pcy + x.pcy * self.ph,
            pw: x.pw * self.pw,
            ph: x.ph * self.ph,
        }
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct DetectionResult {
    pub index: f32,
    pub bbox: BoundingBox,
    pub score: f32,
    pub class: f32,
}

impl DetectionResult {
    #[no_mangle]
    pub extern "C" fn new_detection_result(
        index: f32,
        bbox: BoundingBox,
        score: f32,
        class: f32,
    ) -> DetectionResult {
        DetectionResult { index, bbox, score, class }
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone)]
pub struct DetectionResults(Vec<DetectionResult>);

impl From<Vec<DetectionResult>> for DetectionResults {
    fn from(results: Vec<DetectionResult>) -> Self {
        Self(results)
    }
}

impl Deref for DetectionResults {
    type Target = Vec<DetectionResult>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for DetectionResults {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl DetectionResults {
    pub fn as_f32_slice(&self) -> &[f32] {
        unsafe {
            slice::from_raw_parts(
                self.0.as_ptr() as *const f32,
                self.0.len() * mem::size_of::<DetectionResult>() / mem::size_of::<f32>(),
            )
        }
    }

    pub fn to_query_response(&mut self, id: ResponseId) -> QuerySampleResponse {
        let data = self.0.as_mut_ptr() as usize;
        let size = self.0.len() * mem::size_of::<DetectionResult>();
        QuerySampleResponse { id, data, size }
    }
}

#[inline]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn uninitialized_vec<T>(size: usize) -> Vec<T> {
    let (ptr, _, capacity) = Vec::with_capacity(size).into_raw_parts();
    Vec::from_raw_parts(ptr, size, capacity)
}

#[derive(StructOpt)]
pub enum PostProcessor {
    CppPar,
    RustPar,
    Rust,
}

impl Default for PostProcessor {
    fn default() -> Self {
        PostProcessor::CppPar
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct U8Slice {
    pub ptr: *const u8,
    pub len: usize,
}

impl U8Slice {
    #[no_mangle]
    pub extern "C" fn new_u8_slice(ptr: *const u8, len: usize) -> U8Slice {
        U8Slice { ptr, len }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn do_test_lowered_shape_from_labeled(h: usize, w: usize, c: usize) -> eyre::Result<()> {
        let lowered_shape = LoweredShape::from(&LabeledShape::new_nchw(&[1, c, h, w])?);

        for k in 0..c {
            for i in 0..h {
                for j in 0..w {
                    let labeled_index = k * h * w + i * w + j;
                    let lowered_index = lowered_shape.index(k, i, j);
                    eyre::ensure!(
                        labeled_index == lowered_index,
                        "labeled_index: {}, lowered_index: {}",
                        labeled_index,
                        lowered_index
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn unittest_lowered_shape_from_labeled() -> eyre::Result<()> {
        do_test_lowered_shape_from_labeled(273, 19, 19)
    }

    #[test]
    fn unittest_slice_layout() {
        let v = vec![9u8; 1000];
        let slice = v.as_slice();
        let U8Slice { ptr, len } = unsafe { std::mem::transmute::<_, U8Slice>(slice) };
        assert_eq!(ptr, slice.as_ptr());
        assert_eq!(len, slice.len());
    }
}
