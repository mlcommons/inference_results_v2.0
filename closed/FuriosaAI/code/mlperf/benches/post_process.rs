use criterion::{black_box, criterion_group, criterion_main, Criterion, SamplingMode};

use mlperf::utils::{prepare_raw_output, to_graph, Postprocess};
use mlperf::{ssd_large, ssd_small};

fn do_ssd_small(c: &mut Criterion, group_name: impl Into<String>, remove_unlower: bool) {
    let model_path = "models/ssd_mobilenet_int8.onnx";
    let graph: npu_ir::dfg::Graph =
        to_graph(model_path, 1, remove_unlower).unwrap().try_into().unwrap();
    let pp_seq = ssd_small::Postprocessor::new(&graph);
    let pp_par = pp_seq.clone().with_parallel_processing(true);
    let cpp_par = ssd_small::CppPostprocessor::new(&graph);

    let output =
        prepare_raw_output("coco-300-golden/raw/000000331352.png.raw", graph, model_path).unwrap();
    let data = output.iter().map(|b| b.as_slice::<u8>()).collect::<Vec<_>>();

    let expected = [
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
        81.0,
    ];
    assert_eq!(pp_seq.postprocess(0., &data).as_f32_slice(), expected);
    assert_eq!(pp_par.postprocess(0., &data).as_f32_slice(), expected);
    assert_eq!(cpp_par.postprocess(0., &data).as_f32_slice(), expected);

    let mut group = c.benchmark_group(group_name);
    group
        .sampling_mode(SamplingMode::Flat)
        .sample_size(1000)
        .bench_function("sequential", |b| b.iter(|| pp_seq.postprocess(black_box(0.), &data)))
        .bench_function("parallel", |b| b.iter(|| pp_par.postprocess(black_box(0.), &data)))
        .bench_function("cpp parallel", |b| b.iter(|| cpp_par.postprocess(black_box(0.), &data)));
}

pub fn ssd_small(c: &mut Criterion) {
    do_ssd_small(c, "ssd_small", true)
}

pub fn ssd_small_base(c: &mut Criterion) {
    do_ssd_small(c, "ssd_small_base", false)
}

fn do_ssd_large(c: &mut Criterion, group_name: impl Into<String>, remove_unlower: bool) {
    let model_path = "models/ssd_resnet34_int8.onnx";
    let graph: npu_ir::dfg::Graph =
        to_graph(model_path, 1, remove_unlower).unwrap().try_into().unwrap();
    let pp_seq = ssd_large::Postprocessor::new(&graph);
    let pp_par = pp_seq.clone().with_parallel_processing(true);
    let cpp_par = ssd_large::CppPostprocessor::new(&graph);

    let output =
        prepare_raw_output("coco-1200-golden/raw/000000331352.jpg.raw", graph, model_path).unwrap();
    let data = output.iter().map(|b| b.as_slice::<u8>()).collect::<Vec<_>>();

    let result = pp_seq.postprocess(0., &data);
    assert_eq!(
        result.as_f32_slice(),
        ssd_large::value::RESULT_000000331352,
        "# of detections: {}",
        result.as_f32_slice().len()
    );

    let result = pp_par.postprocess(0., &data);
    assert_eq!(result.as_f32_slice(), ssd_large::value::RESULT_000000331352,);

    let result = cpp_par.postprocess(0., &data);
    assert_eq!(result.as_f32_slice(), ssd_large::value::RESULT_000000331352,);

    let mut group = c.benchmark_group(group_name);
    group
        .sampling_mode(SamplingMode::Flat)
        .sample_size(1000)
        .bench_function("sequential", |b| b.iter(|| pp_seq.postprocess(black_box(0.), &data)))
        .bench_function("parallel", |b| b.iter(|| pp_par.postprocess(black_box(0.), &data)))
        .bench_function("cpp parallel", |b| b.iter(|| cpp_par.postprocess(black_box(0.), &data)));
}

pub fn ssd_large(c: &mut Criterion) {
    do_ssd_large(c, "ssd_large", true)
}

pub fn ssd_large_base(c: &mut Criterion) {
    do_ssd_large(c, "ssd_large_base", false)
}

criterion_group!(benches, ssd_small, ssd_small_base, ssd_large, ssd_large_base);
criterion_main!(benches);
