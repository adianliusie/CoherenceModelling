from torch.optim.lr_scheduler import LambdaLR
import torch
import torch.nn as nn
import numpy as np

from .utils import Batcher, DataHandler, TokenizerClass
from .models import BilstmHier, TransformerFlat

class log_sigmoid_loss(nn.Module):
        def __init__(self):
            super().__init__()
            self.log_sigmoid = nn.LogSigmoid()

        def forward(self, inputs):
            log_likelihood = self.log_sigmoid(inputs)
            loss =  -1 * torch.mean(log_likelihood)
            return loss

class ExperimentHandler:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.cross_loss = nn.CrossEntropyLoss()
        self.log_sigmoid_loss = log_sigmoid_loss()

    def train(self, config):
        D = DataHandler(config.data_src)
        T = TokenizerClass(config.system, config.embed_lim)
        B = Batcher(config.bsz, config.schemes, config.args, config.max_len, T)
        
        self.model = TransformerFlat(config.system, config.attention)
        #self.model = BilstmHier(T.embeddings)
        model = self.model
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        if config.scheduling:
              SGD_steps = (len(train_data)*config.epochs)/self.bsz
              lambda1 = lambda i: 10*i/SGD_steps if i <= SGD_steps/10 else 1 - ((i - 0.1*SGD_steps)/(0.9*SGD_steps))
              scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
        
        model.to(self.device)
        B.to(self.device) 

        for epoch in range(config.epochs):
            #Training
            logger = np.zeros(3)
            for k, batch in enumerate(B.make_batches(D.train, config.c_num, config.hier)):
                if config.hier:  loss, batch_acc = self.calc_batch_hier(model, batch)
                else:            loss, batch_acc = self.calc_batch(model, batch)
                logger += [loss.item(), *batch_acc]
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if config.scheduling: scheduler.step()

                if k%config.debug_sz==0 and k!=0:
                    print(f'{epoch:<2} {k:<6} {logger[0]/config.debug_sz:.3f}   {logger[1]/logger[2]:.4f}')
                    logger = np.zeros(3)
 
            #Dev
            logger = np.zeros(3)
            for k, batch in enumerate(B.make_batches(D.dev, config.c_num, config.hier)):
                if config.hier:  loss, batch_acc = self.calc_batch_hier(model, batch, no_grad=True)
                else:            loss, batch_acc = self.calc_batch(model, batch, no_grad=True)
                logger += [loss.item(), *batch_acc]
            
            print(f'\n{len(D.dev):<5} {logger[0]/k:.3f}   {logger[1]/logger[2]:.4f}\n', 50*'--')
                
    def calc_batch_hier(self, model, batch, no_grad=False):
        if no_grad==True:
            with torch.no_grad():
                return self.calc_batch_hier(model, batch, no_grad=False)
            
        loss, acc = 0, np.zeros(2)
        for pos, neg in batch:
            y_pos = model(pos.ids, pos.mask)
            y_neg = model(neg.ids, neg.mask)
            loss += self.log_sigmoid_loss(y_pos - y_neg)/len(batch)
            acc += [(y_pos>y_neg).item(), 1]
        return loss, acc

    def calc_batch(self, model, batch, no_grad=False):
        if no_grad==True:
            with torch.no_grad():
                return self.calc_batch(model, batch, no_grad=False)
            
        pos, neg = batch
        y_pos = model(pos.ids, pos.mask)
        y_neg = model(neg.ids, neg.mask)
        loss = self.log_sigmoid_loss(y_pos - y_neg)
        acc = [sum(y_pos - y_neg > 0).item(), len(y_pos)]
        return loss, acc