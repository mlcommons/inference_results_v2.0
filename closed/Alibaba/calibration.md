## MLPerf Inference v2.0 - Calibration
We have two types of calibration for the MLPerf submissions. One is for Panjiu-M System and the other one is for Haishen System (with Qualcomm AIC100 accelerator ).

### MLPerf Inference v2.0 - Panjiu-M - Calibration
  
For Panjiu-M system, our submission utilizes FP16 precision , therefore needs no INT8/UINT8 quantization nor calibration.
As for the FP32 to FP16 conversion,  the process is done via [```tvm.relay.transform.ToMixedPrecision('float16')(...)```](https://tvm.apache.org/docs/reference/api/python/relay/transform.html?highlight=tomixedprecision#tvm.relay.transform.ToMixedPrecision);

### MLPerf Inference v2.0 - Haishen - Calibration

For Haishen system with Qualcomm AIC100 accelerator, we use regular profile-guided post-training quantization.
We pass a set of calibration samples through the neural network to obtain a
profile of tensor values for the network operations.  We then use the profile
to calculate the scales and offsets for quantization to have a negligible
impact on accuracy.
