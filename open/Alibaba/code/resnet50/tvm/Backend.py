import os
import sys
import logging
import tvm
import onnx
import tvm.relay as relay
from baseBackend import baseBackend
from tvm.contrib import graph_executor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("BACKEND")


class Backend(baseBackend):
    def __init__(self, model_param, dataset_param):
        self.batch_size = dataset_param["batch_size"]
        self.image_size = dataset_param["image_size"]
        self.precision = dataset_param["precision"]
        self.model_path = model_param["model_path"]
        self.input_layer_name = model_param["input_layer_name"]
        self.output_layer_name = model_param["output_layer_name"]
        self.tvm_llvm_target = model_param["tvm_llvm_target"]
        self.layout = dataset_param["layout"]
        self.tvm_opt_level = model_param["tvm_opt_level"]
        self.num_cls = model_param["num_cls"]
        self.output_shape = (self.batch_size, self.num_cls)
        if not os.path.isfile(self.model_path):
            log.error("Model not found: {}".format(self.model_path))
            sys.exit(1)
        print("Loaded pretrained model")

    def load_model(self):
        print("model_path: " + self.model_path)
        target = self.tvm_llvm_target
        dev = tvm.device(str(target), 0)
        if self.layout == "NCHW":
            shape_dict = {self.input_layer_name: [self.batch_size, 3, self.image_size, self.image_size]}
        else:
            shape_dict = {self.input_layer_name: [self.batch_size, self.image_size, self.image_size, 3]}

        if self.model_path.endswith("onnx"):
            onnx_model = onnx.load(self.model_path)
            mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
            with tvm.transform.PassContext(opt_level=self.tvm_opt_level):
                lib = relay.build(mod, target=target, params=params)
            self.model = graph_executor.GraphModule(lib["default"](dev))

        elif self.model_path.endswith("so"):
            lib = tvm.runtime.load_module(self.model_path)
            self.model = graph_executor.GraphModule(lib["default"](dev))
        elif self.model_path.endswith("tflite"):
            assert self.layout == "NHWC"
            tflite_model_buf = open(self.model_path, "rb").read()
            try:
                import tflite
                tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
            except AttributeError:
                import tflite.Model
                tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
            mod, params = relay.frontend.from_tflite(tflite_model,
                                                     shape_dict,
                                                     dtype_dict={self.input_layer_name: self.precision})
            with tvm.transform.PassContext(opt_level=self.tvm_opt_level):
                lib = relay.build(mod, target=target, params=params)
            self.model = graph_executor.GraphModule(lib["default"](dev))
        log.info("Model loaded")

    def predict(self, data):
        self.model.set_input(self.input_layer_name, data)
        self.model.run()
        tvm_output = self.model.get_output(0, tvm.nd.empty(self.output_shape, dtype=self.precision)).numpy()

        return tvm_output
