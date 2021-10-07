import torch.nn as nn
import torch 

from .model_utils import Attention

class BilstmHier(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = nn.Embedding(len(embeddings), 300)
        #self.embeddings = nn.Embedding.from_pretrained(embeddings)
        #self.dropout = nn.Dropout(p=0.5)
        self.bilstm_1 = nn.LSTM(input_size=300, hidden_size=150, num_layers=1, bias=True,
                                batch_first=True, dropout=0, bidirectional=True)
        self.bilstm_2 = nn.LSTM(input_size=300, hidden_size=150, num_layers=1, bias=True, 
                                batch_first=True, dropout=0, bidirectional=True)
        self.attent_1 = Attention(300)
        self.attent_2 = Attention(300)
        self.classifier = nn.Linear(300, 1)

    def forward(self, x, mask):
        x = self.embeddings(x)
        #x = self.dropout(x)
        mask_lens = torch.sum(mask, dim=-1).cpu()
        x_padded = torch.nn.utils.rnn.pack_padded_sequence(x, mask_lens, batch_first=True, enforce_sorted=False)
        H1_padded, _ = self.bilstm_1(x_padded) 
        H1, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(H1_padded, batch_first=True)
        h1 = self.attent_1(H1, mask)
        H2, _ = self.bilstm_2(h1.unsqueeze(0)) 
        h2 = self.attent_1(H2)
        y = self.classifier(h2).squeeze(0).squeeze(-1)
        return y


