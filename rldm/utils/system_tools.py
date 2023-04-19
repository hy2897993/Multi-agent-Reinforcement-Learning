import os

import torch


def get_cpu_gpu_count():    
    n_cpu = os.cpu_count()
    has_gpu = torch.cuda.is_available()
    n_gpu = 0 if not has_gpu else torch.cuda.device_count()
    return n_cpu, n_gpu
