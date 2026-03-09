import torch

class AverageMeter:

    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    def reset(self):
        self.value = 0.0
        self.sum = 0.0
        self.average = 0.0
        self.count = 0
    
    def update(self, value: float, n: int=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.average:.5f}"
    
def clear_gpu_cache():
    """Aggressively clear GPU cache and synchronize"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()