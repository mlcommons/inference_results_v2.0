# MLPerf™ Deci-AI Bert-99 Results  

The results:

| Submitter  | System |	Processor   | Software  	| Offline (samples/s) |	SQuAD V1  F1 Score |
|------------|--------|-------------|-----------------------|---------------------|----------------------------|
| Deci-AI    | Amazon EC2 (m5dn.8xlarge) | Intel(R) Xeon(R) Platinum 8259CL CPU @ 2.50GHz | OpenVino  | 44 | 89.9 |
| Deci-AI    | Amazon EC2 (c6i.16xlarge) | Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz | OpenVino  | 121 | 89.9 |

## Reproducibility
Due to intellectual property issues we share our benchmarking code but do no disclosed our models and inference parallization package. 
To have an interactive session to reproduce the results contact shai.rozenberg@deci.ai

### Requiremnts
```
pip install inference/requirements.txt
```

### Offline

```
python run.py --backend=openvino --scenario=Offline --model_path openvino_models/DeciBERT.pkl --tokenizer deci --batch_size 16 --log_path /inference/language/bert/results/ --use_infery_pro --infery_pro_return_cycle 800 --num_inferencers 2
```
### Server
```
python run.py --backend=openvino --scenario=Server --model_path openvino_models/DeciBERT.pkl --tokenizer deci --log_path /inference/language/bert/results/ 
```
