"""
OpenVINO backend (https://github.com/openvinotoolkit/openvino)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

import onnxruntime as rt
import infery  # Deci's python runtime inference engine (https://pypi.org/project/infery)
import backend


class BackendOpenVino(backend.Backend):
    def __init__(self):
        super(BackendOpenVino, self).__init__()

    def version(self):
        return rt.__version__

    def name(self):
        """Name of the runtime."""
        return "onnxruntime"

    def image_format(self):
        """image_format. For onnx it is always NCHW."""
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        """Load model and find input/outputs from the model file."""
        self.sess = infery.load(model_path=model_path, framework_type='openvino', inference_hardware='cpu')
        # get input and output names
        if not inputs:
            self.inputs = ['input']
        else:
            self.inputs = inputs
        if not outputs:
            self.outputs = ['output']
        else:
            self.outputs = outputs
        return self

    def predict(self, feed):
        """Run the prediction."""
        return self.sess.predict(feed['input'])
