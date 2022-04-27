# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5(OfflineGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_R5300G5
    run_infer_on_copy_streams = True
    gpu_batch_size = 1024           
    gpu_copy_streams = 4
    gpu_inference_streams = 1      
    offline_expected_qps = 150000   
    use_graphs = True            


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_R5300G5_Triton(A30X8_CUSTOM_R5300G5):
    use_triton = True
    gpu_batch_size = 2048      
    gpu_copy_streams = 4
    run_infer_on_copy_streams = True
    use_graphs = False
    offline_expected_qps = 151200.0       
    




