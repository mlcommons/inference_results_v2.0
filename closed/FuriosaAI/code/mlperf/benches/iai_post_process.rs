use iai::black_box;

use mlperf::utils::{prepare_raw_output, to_graph, Postprocess};
use mlperf::{ssd_large, ssd_small};

fn ssd_small_postprocess_sequential() {
    let model_path = "models/ssd_mobilenet_int8.onnx";
    let graph: npu_ir::dfg::Graph = to_graph(model_path, 1, true).unwrap().try_into().unwrap();
    let pp = ssd_small::Postprocessor::new(&graph);

    let output =
        prepare_raw_output("coco-300-golden/raw/000000331352.png.raw", graph, model_path).unwrap();
    let data = output.iter().map(|b| b.as_slice::<u8>()).collect::<Vec<_>>();

    pp.postprocess(black_box(0.), &data);
}

fn ssd_small_postprocess_parallel() {
    let model_path = "models/ssd_mobilenet_int8.onnx";
    let graph: npu_ir::dfg::Graph = to_graph(model_path, 1, true).unwrap().try_into().unwrap();
    let pp = ssd_small::Postprocessor::new(&graph).with_parallel_processing(true);

    let output =
        prepare_raw_output("coco-300-golden/raw/000000331352.png.raw", graph, model_path).unwrap();
    let data = output.iter().map(|b| b.as_slice::<u8>()).collect::<Vec<_>>();

    pp.postprocess(black_box(0.), &data);
}

fn ssd_large_postprocess_sequential() {
    let model_path = "models/ssd_resnet34_int8.onnx";
    let graph: npu_ir::dfg::Graph = to_graph(model_path, 1, true).unwrap().try_into().unwrap();
    let pp = ssd_large::Postprocessor::new(&graph);

    let output =
        prepare_raw_output("coco-1200-golden/raw/000000331352.jpg.raw", graph, model_path).unwrap();
    let data = output.iter().map(|b| b.as_slice::<u8>()).collect::<Vec<_>>();

    pp.postprocess(black_box(0.), &data);
}

iai::main!(
    ssd_small_postprocess_sequential,
    ssd_small_postprocess_parallel,
    ssd_large_postprocess_sequential,
);
