# MLPerf™ Deci-AI BERT Submission

## Reproducibility
Due to intellectual property issues we share our benchmarking code but do no disclosed our models and inference parallization package. 
To have an interactive session to reproduce the results contact shai.rozenberg@deci.ai

## Model Details
All models were compiled using OpenVINO and quantized to INT8.

### Requiremnts
```
pip install inference/requirements.txt
```

### Offline

```
python run.py --backend=openvino --scenario=Offline --model_path openvino_models/DeciBERT.pkl --tokenizer deci --batch_size <BATCH_SIZE> --log_path <OUTPUT_PATH> --use_infery_pro --num_inferencers 2
```
