import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, RobertaModel, ElectraModel

from .model_utils import Attention, get_transformer

class DocumentClassifier(nn.Module):
    def __init__(self, class_num=1, system=None, hier=None, embeds=None):  
        super().__init__()
        self.classifier = nn.Linear(300, class_num)
        
        if system in ['bert','roberta','electra']: self.sent_encoder = TransEncoder(system) 
        elif system in ['glove', 'word2vec']:      self.sent_encoder = BilstmEncoder(embeds) 

        self.hier = hier
        if hier == 'transformer': self.doc_encoder = HierTransEncoder(hier)
        elif hier == 'bilstm': self.doc_encoder = HierTransEncoder(hier)

    def forward(self, x, mask):
        H1 = self.sent_encoder(x, mask)
        if self.hier:
            y = y.unsqueeze(0)
            y = self.doc_encoder(y)
        y = self.classifier(y)
        return y

class TransEncoder(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.transformer = get_transformer(name)
        
    def forward(self, x, mask):
        H1 = self.transformer(input_ids=x, attention_mask=mask).last_hidden_state
        return H1

class BilstmEncoder(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.bilstm = nn.LSTM(input_size=300, hidden_size=150, num_layers=2, 
                                    bias=True, batch_first=True, dropout=0, bidirectional=True)
        self.embeddings = nn.Embedding.from_pretrained(embeddings)
        
    def forward(self, x, mask):
        x = self.embeddings(x)
        mask_lens = torch.sum(mask, dim=-1).cpu()
        x_padded = torch.nn.utils.rnn.pack_padded_sequence(x, mask_lens, batch_first=True, enforce_sorted=False)
        output, _ = self.bilstm(x_padded) 
        H1, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return H1

class HierTransEncoder(nn.Module):
    def __init__(self, name, hsz=300):
        super().__init__()
        config = BertConfig(hidden_size=hsz, num_hidden_layers=12, num_attention_heads=12, intermediate_size=4*hsz)
        self.transformer = BertModel(config)

    def forward(self, x):
        hidden_vectors = self.transformer(inputs_embeds=x, return_dict=True).last_hidden_state[:,0]
        return hidden_vectors 

class HierBilstmEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bilstm = nn.LSTM(input_size=300, hidden_size=150, num_layers=1, bias=True, 
                              batch_first=True, dropout=0, bidirectional=True)
        
    def forward(self, x):  
        h1, _ = self.bilstm(x) 
        h1 = torch.mean(h1, dim=1)
        return output




