from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
import numpy as np

class ReimbursementModel:
    def __init__(self):
        # Input features: trip_duration_days, miles_traveled, total_receipts_amount
        self.l1 = Linear(3, 64)
        self.l2 = Linear(64, 128)
        self.l3 = Linear(128, 64)
        self.l4 = Linear(64, 1)
        
    def __call__(self, x):
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        x = self.l3(x).relu()
        x = self.l4(x)
        return x
    
    def save(self, path):
        params = []
        for layer in [self.l1, self.l2, self.l3, self.l4]:
            params.extend([layer.weight.numpy(), layer.bias.numpy()])
        np.save(path, np.array(params, dtype=object), allow_pickle=True)
    
    def load(self, path):
        params = np.load(path, allow_pickle=True)
        for i, layer in enumerate([self.l1, self.l2, self.l3, self.l4]):
            layer.weight = Tensor(params[i*2])
            layer.bias = Tensor(params[i*2+1]) 
