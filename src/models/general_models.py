import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, RobertaModel, ElectraModel

from ..utils.tokenizer import get_embeddings
from .model_utils import Attention, get_transformer 

class DocumentClassifier(nn.Module):
    def __init__(self, config):  
        super().__init__()
        self.classifier = nn.Linear(300, 1)
        
        if config.system in ['bert','roberta','electra']: self.first_encoder = TransEncoder(config.system) 
        elif config.system in ['glove', 'word2vec']:      self.first_encoder = BilstmEncoder(config.system) 

        if config.pooling == 'attention':
            self.attention = Attention(300)
            self.pooling = lambda ids, mask=None: self.attention(ids, mask)
        elif config.pooling == 'first':
            self.pooling = lambda ids, mask: ids[:,0]
        
        self.hier = config.hier
        if self.hier:
            if config.system in ['bert','roberta','electra']: self.second_encoder = HierTransEncoder(300) 
            elif config.system in ['glove', 'word2vec']:      self.second_encoder = HierBilstmEncoder(300) 
        
            if config.pooling == 'attention':
                self.attention_2 = Attention(300)
                self.pooling_2 = lambda ids: self.attention(ids)
            elif config.pooling == 'first':
                self.pooling_2 = lambda ids: ids[:,0]
            
    def forward(self, x, mask):
        H1 = self.first_encoder(x, mask)
        h = self.pooling(H1, mask)
        if self.hier:
            h = h.unsqueeze(0)
            H2 = self.second_encoder(h)
            h = self.pooling_2(H2)
        y = self.classifier(h).squeeze(-1)
        return y

class TransEncoder(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.transformer = get_transformer(name)
        self.linear = nn.Linear(768, 300)

    def forward(self, x, mask):
        H1 = self.transformer(input_ids=x, attention_mask=mask).last_hidden_state
        H1 = self.linear(H1)
        return H1

class BilstmEncoder(nn.Module):
    def __init__(self, system):
        super().__init__()
        self.bilstm = nn.LSTM(input_size=300, hidden_size=150, num_layers=2, 
                                    bias=True, batch_first=True, dropout=0, bidirectional=True)
        embeddings = get_embeddings(system)
        self.embeddings = nn.Embedding.from_pretrained(embeddings)
        
    def forward(self, x, mask):
        x = self.embeddings(x)
        mask_lens = torch.sum(mask, dim=-1).cpu()
        x_padded = torch.nn.utils.rnn.pack_padded_sequence(x, mask_lens, batch_first=True, enforce_sorted=False)
        output, _ = self.bilstm(x_padded)
        H1, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return H1

class HierTransEncoder(nn.Module):
    def __init__(self, hsz=300):
        super().__init__()
        heads = hsz//64
        config = BertConfig(hidden_size=hsz, num_hidden_layers=6, num_attention_heads=heads, intermediate_size=4*hsz)
        self.transformer = BertModel(config)

    def forward(self, x):
        H1 = self.transformer(inputs_embeds=x, return_dict=True).last_hidden_state
        return H1 

class HierBilstmEncoder(nn.Module):
    def __init__(self, hsz=300):
        super().__init__()
        self.bilstm = nn.LSTM(input_size=hsz, hidden_size=hsz//2, num_layers=1, bias=True, 
                              batch_first=True, dropout=0, bidirectional=True)
        
    def forward(self, x):
        H1, _ = self.bilstm(x)
        return H1




