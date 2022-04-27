# MLPerf™ Deci-AI Image Classification Results  
The code is based on the reference code provided in https://github.com/mlcommons/inference/tree/r2.0 
with the addition of the OpenVINO backend and the inference package 'infery'


## The results:

| Submitter  | System |	Processor   | Software  	| Offline (samples/s) |	ImageNet Eval-Top1-Accuracy |
|------------|--------|-------------|--------------------|---------------------|----------------------------|
| Deci-AI    | Amazon EC2 (m5zn.6xlarge) | Intel(R) Xeon(R) Platinum 8252C CPU @ 3.80GHz | OpenVino  | 2270.13 | 78.418% |
| Deci-AI    | Amazon EC2 (c6i.16xlarge) | Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz | OpenVino  | 4313.98 | 78.418% |
| Deci-AI    | Amazon EC2 (c6i.2xlarge)  | Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz | OpenVino  | 1207.45  | 78.418% |

## Reproducibility
Due to intellectual property issues we share our benchmarking code and do no disclosed our models. 
To have an interactive session to reproduce the results contact shai.rozenberg@deci.ai

### Requiremnts
```
pip install requirements.txt
```

### Offline

```
python -u main.py --scenario Offline --dataset imagenet_pytorch --profile resnet50-openvino --mlperf_conf mlperf.conf --model DeciNetCPU.pkl --dataset-path <IMAGENET_PATH> --output <OUTPUT_PATH> --user_conf user.conf --max-batchsize 32
```
