import torch
import torch.nn as nn
import math

# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev, dtype=torch.float64)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
            
        inp = inp.type(torch.float64)
        
        # (A) 먼저 inp 자체에 NaN/Inf가 있는지 검사
        if torch.isnan(inp).any() or torch.isinf(inp).any():
            print(">>> Detected NaN or Inf in `inp` before updating scaler_row!")
            print("inp:", inp)
            return
        
        # (B) 업데이트 직전 self.scaler_row 검사
        if torch.isnan(self.scaler_row).any() or torch.isinf(self.scaler_row).any():
            print(">>> Detected NaN or Inf in `self.scaler_row` before update!")
            print("self.scaler_row:", self.scaler_row)
            return
        
        if self.nsamples == 0:
            self.nsamples = tmp
            self.scaler_row = torch.norm(inp, p=2, dim=1) ** 2
        else:
            ratio = self.nsamples / (self.nsamples + tmp)
            if ratio < 0 or ratio > 1 or math.isnan(ratio):
                print(f">>> Invalid ratio: {ratio}, nsamples={self.nsamples}, tmp={tmp}")
                return
            self.scaler_row *= ratio
            self.nsamples += tmp
            
            self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
            
            # (D) 업데이트 후 검사
        if torch.isnan(self.scaler_row).any() or torch.isinf(self.scaler_row).any():
            print(">>> Detected NaN or Inf in `self.scaler_row` after update!")
            print("self.scaler_row:", self.scaler_row)
            print(f"nsamples={self.nsamples}, tmp={tmp}")
            return
        