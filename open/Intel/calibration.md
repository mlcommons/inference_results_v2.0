# Intel's BootstrapNAS MLPerf Submission


## BootstrapNAS​

BootstrapNAS is a modular, multi-stage software AI automation capability that takes as input a pre-trained DL model, internally it automatically generates a super-network, and performs a Neural Architecture Search (NAS), producing as result, a set of efficient ready-to-deploy models that outperform the original model in terms of either accuracy, performance, or both objectives for the underlying target hardware platform. This capability is under development and will soon be open sourced.

<p align="center">
<img src="../../../architecture.png" alt="BootstrapNAS Architecture" width="500"/>
</p>

```BibTex
@article{DBLP:journals/corr/abs-2112-10878,
  author    = {J. Pablo Muñoz and Nikolay Lyalyushkin and Yash Akhauri and Anastasia Senina and
               Alexander Kozlov and Nilesh Jain},
  title     = {Enabling NAS with Automated Super-Network Generation},
  journal   = {CoRR},
  volume    = {abs/2112.10878},
  year      = {2021},
  url       = {https://arxiv.org/abs/2112.10878},
  eprinttype = {arXiv},
  eprint    = {2112.10878},
  timestamp = {Tue, 04 Jan 2022 15:59:27 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2112-10878.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


## Generate IR for BootstrapNAS

The BootstrapNAS models were generated and optimized following these steps:
1. Starting from the FP32 Torchvision ResNet-50 [model](https://download.pytorch.org/models/resnet50-0676ba61.pth), use BootstrapNAS automation capabilities to generate a weight-sharing super-network, a.k.a. *OFA* (Cai et al. 2020) or *Single Stage* (Yu et al. 2020).
2. Train the super-network and apply a hardware-aware search to identify optimal sub-networks for the target hardware. Approach is described in our [paper](https://arxiv.org/abs/2112.10878). 
3. Quantize (8-bit) the subnetworks using the [Neural Network Compression Framework](https://github.com/openvinotoolkit/nncf) (NNCF).
4. Optimize the quantized model using OpenVINO's model optimizer.
​

## Model Quantization 

The models were quantized (8-bit) using the following configuration for NNCF:
```python
model = torch.load(<SUBNETWORK_MODEL_PATH>)
nncf_config = {
    'input_info': {
        'sample_size': [1, 3, 224, 224]
    },
    "device": 'cuda:0',
    "compression": {
        "algorithm": "quantization",
        "initializer": {
            "range": {
                "num_init_samples": 850
            }
        }
    }
}

nncf_config = NNCFConfig.from_dict(nncf_config)
nncf_config = register_default_init_args(
nncf_config, data_loader, device=nncf_config.get('device'))
compression_ctrl, model = create_compressed_model(model, nncf_config)
```
More information about NNCF's quantization approach can be found [here](https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Quantization.md).


## Model Optimization and Generation of OpenVINO IR files
​
The quantized models were further optimized using OpenVINO's [Model Optimizer](https://docs.openvino.ai/2021.2/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) which is called from OpenVINO's Accuracy Checker.
For this step, we use the following command to generate the IR files:

```
AC_DIR=<OPENVINO_DIR>/open_model_zoo/tools/accuracy_checker

accuracy_check \
-c <CONFIGURATION.yml> \ 
-m $DIR \
-s <IMAGENET_DIR> \
-a <OPENVINO_DIR>/accuracy_checker_annotations/2012 \
-d $AC_DIR/dataset_definitions.yml
```


The contents of the configuration YAML file has the following parameters and flags:
```yaml
models: 
  - name: resnet50_imagenet
    launchers:
      - framework: dlsdk
        device: CPU
        adapter: classification
        onnx_model: <input_model>
        mo_params:
            output: <Model Name>
            mean_values: "(123.675, 116.28,103.53)"
            scale_values: "(58.395, 57.12 , 57.375)"
            input_shape: "(1,3,224,224)"
        
        mo_flags:
            - reverse_input_channels
        
        datasets:
            - name: imagenet_1000_classes
              data_source: val
              annotation: imagenet1000.pickle
              reader: pillow_imread
        
              preprocessing:
              - type: resize
                size: 256
                aspect_ratio_scale: greater
                use_pillow: True
                interpolation: BILINEAR
              - type: crop
                size: 224
                use_pillow: True
              - type: bgr_to_rgb
        
              metrics:
                - name: accuracy@top1
                  type: accuracy
                  top_k: 1
```
