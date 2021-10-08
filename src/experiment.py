from torch.optim.lr_scheduler import LambdaLR
import torch
import torch.nn as nn
import numpy as np
from scipy import stats
import os
import json 
import random 
from tqdm import tqdm

from .utils import Batcher, DataHandler
from .models import DocumentClassifier

class log_sigmoid_loss(nn.Module):
        def __init__(self):
            super().__init__()
            self.log_sigmoid = nn.LogSigmoid()

        def forward(self, inputs):
            log_likelihood = self.log_sigmoid(inputs)
            loss =  -1 * torch.mean(log_likelihood)
            return loss

        def ranking(self, y_pos, y_neg):
            loss = self.forward(y_pos - y_neg)
            return loss
        
        def classification(self, y_pos, y_neg):
            loss = (self.forward(y_pos) + self.forward(-1*y_neg)) /2
            return loss

def save_json(path, data):
    with open(path, 'x') as jsonFile:
        json.dump(data, jsonFile)
        jsonFile.close()

def set_up_logger(log_path):
    def printf(string):
        with open(log_path, 'a+') as f:
            f.write(string + '\n')
            print(string)
    return printf

class ExperimentHandler:
    def __init__(self):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() \
                 else torch.device('cpu')
        self.cross_loss = nn.CrossEntropyLoss()
        self.log_sigmoid_loss = log_sigmoid_loss()

    def train_corruption(self, config):
        print(f'Parameters: {config}')
        
        ## First part of the script sets up all the helpers
        D = DataHandler(config.data_src)
        B = Batcher(config.system, config.bsz, config.schemes, 
                    config.args, config.max_len) 

        if config.debug_cut: 
            D.train = D.train[:config.debug_cut]
            
        self.model = DocumentClassifier(config)
        model = self.model
        model.to(self.device)
        model.train()
        B.to(self.device) 

        if config.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        elif config.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=config.lr)
        
        if config.scheduling:
              SGD_steps = (len(train_data)*config.epochs)/self.bsz
              lambda1 = lambda i: 10*i/SGD_steps if i <= SGD_steps/10 else 1 - ((i - 0.1*SGD_steps)/(0.9*SGD_steps))
              scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
        
        if config.loss == 'ranking':
            self.loss_function = self.log_sigmoid_loss.ranking
        elif config.loss == 'classification':
            self.loss_function = self.log_sigmoid_loss.classification

        if config.save:
            save_dir = f'/home/alta/Conversational/OET/al826/2021/coherence/results/framework_1/{config.model_name}'
            os.mkdir(save_dir)
            save_json(path=f'{save_dir}/config.json', data=config._asdict())
            self.printf = set_up_logger(f'{save_dir}/log.txt')
            best_epoch = [-1, 0]
            
        print('BEGINNING TRAINING: ', int(len(D.train)*config.c_num/config.bsz), 'BATCHES PER EPOCH')
        for epoch in range(config.epochs):
            #Training
            logger = np.zeros(3)
            for k, batch in enumerate(B.batches(D.train, config.c_num, config.hier)):
                if config.hier:  loss, batch_acc = self.calc_batch_hier(model, batch)
                else:            loss, batch_acc = self.calc_batch(model, batch)
                logger += [loss.item(), *batch_acc]
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if config.scheduling: scheduler.step()

                if k%config.debug_sz==0 and k!=0:
                    self.printf(f'{epoch:<2} {k:<6} {logger[0]/config.debug_sz:.3f}   {logger[1]/logger[2]:.4f}')
                    logger = np.zeros(3)

            #Dev
            logger = np.zeros(3)
            for k, batch in enumerate(B.batches(D.dev, config.c_num, config.hier)):
                if config.hier:  loss, batch_acc = self.calc_batch_hier(model, batch, no_grad=True)
                else:            loss, batch_acc = self.calc_batch(model, batch, no_grad=True)
                logger += [loss.item(), *batch_acc]
            
            dev_loss, dev_acc = logger[0]/k, logger[1]/logger[2]
            
            self.printf(f'\n DEV  {dev_loss:.3f}   {dev_acc:.4f}\n')

            #Save model
            if config.save:
                if dev_acc > best_epoch[1]:
                    self.printf(f'SAVING MODEL AT EPOCH {epoch}')
                    model.to("cpu")
                    torch.save(model.state_dict(), f'{save_dir}/model.pt')
                    model.to(self.device)
                    best_epoch[1] = dev_acc 
               
            #GCDC correlation
            self.eval_GCDC(config)
            
            self.printf(50*'--')

    def calc_batch_hier(self, model, batch, no_grad=False):
        if no_grad==True:
            with torch.no_grad():
                return self.calc_batch_hier(model, batch, no_grad=False)
            
        loss, acc = 0, np.zeros(2)
        for pos, neg in batch:
            y_pos = model(pos.ids, pos.mask)
            y_neg = model(neg.ids, neg.mask)
            loss += self.loss_function(y_pos, y_neg)/len(batch)
            acc += [(y_pos>y_neg).item(), 1]
        return loss, acc

    def calc_batch(self, model, batch, no_grad=False):
        if no_grad==True:
            with torch.no_grad():
                return self.calc_batch(model, batch, no_grad=False)

        pos, neg = batch
        y_pos = model(pos.ids, pos.mask)
        y_neg = model(neg.ids, neg.mask)

        loss = self.loss_function(y_pos, y_neg)
        acc = [sum(y_pos - y_neg > 0).item(), len(y_pos)]
        return loss, acc

    def eval_corruption(self, config):
        random.seed(10)
        D = DataHandler(config.data_src)
        B = Batcher(config.system, config.bsz, config.schemes, 
                    config.args, config.max_len) 
        
        model = DocumentClassifier(config)
        PATH = f'/home/alta/Conversational/OET/al826/2021/coherence/results/framework_1/{config.model_name}/model.pt'
        model.load_state_dict(torch.load(PATH))
        model.to(self.device)
        model.eval()
        B.to(self.device)

        logger = np.zeros(3)
        with torch.no_grad():
            for k, batch in enumerate(tqdm(B.batches(D.test, c_num=20, hier=config.hier))):
                if config.hier:  loss, batch_acc = self.calc_batch_hier(model, batch, no_grad=True)
                else:            loss, batch_acc = self.calc_batch(model, batch, no_grad=True)
                logger += [loss.item(), *batch_acc]

        self.printf('FINAL EVAL')
        self.printf(f'\n{len(D.test):<5} {logger[0]/k:.3f}   {logger[1]/logger[2]:.4f}\n')
        
    def eval_GCDC(self, config):
        D = DataHandler('gcdc')
        B = Batcher(config.system, config.bsz, config.schemes, 
                    config.args, config.max_len) 
        model = self.model
        model.to(self.device)
        B.to(self.device)

        scores = []
        predictions = []

        for k, batch in enumerate(B.labelled_batches(D.clinton_train, config.hier)):
            if config.hier:
                for doc in batch:
                    y = model(doc.ids, doc.mask)
                    predictions.append(y.item())
                    scores.append(doc.score.item())
            else:
                y = model(batch.ids, batch.mask)
                predictions += y.tolist()
                scores += batch.score.tolist()
                
        self.get_correlations(predictions, scores)
        
    def get_correlations(self, predictions, scores):
        pearson = stats.pearsonr(predictions, scores)[0]
        spearman = stats.spearmanr(predictions, scores)[0]
        self.printf(f'GCDC correlation: PEAR {pearson:.3f}    SPEAR P {spearman:.3f}')

        
        
