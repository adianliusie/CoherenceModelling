import torch
import torch.nn as nn

from .model_utils import Attention, get_transformer

class TransformerFlat(nn.Module):
    def __init__(self, system='bert', attention=False):  
        super().__init__()    
        self.transformer = get_transformer(system) 
        self.classifier = nn.Linear(768, 1)
        if attention: self.attention = Attention(768)
            
    def forward(self, x, mask):
        H1 = self.transformer(x, mask)
        if hasattr(self, 'attention'):
            h1 = self.attention(H1, mask)
        else:
            h1 = H1.last_hidden_state[:,0]
        y = self.classifier(h1)
        return y

