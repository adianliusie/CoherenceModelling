from torch.optim.lr_scheduler import LambdaLR
import torch
import torch.nn as nn
import numpy as np
from scipy import stats
import random 
from tqdm import tqdm

from .utils import Batcher, DataHandler, Logger
from .utils.config import select_optimizer, select_loss
from .models import DocumentClassifier


class ExperimentHandler:
    def __init__(self):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() \
                      else torch.device('cpu')
        self.cross_loss = nn.CrossEntropyLoss()

    def train_corruption(self, config):
        print(f'Parameters: {config}')
        
        # First part of the script is set up
        L = Logger(config)
        D = DataHandler(config.data_src)
        B = Batcher(config.system, config.bsz, config.schemes, 
                    config.args, config.max_len) 
        
        if config.debug_len: 
            D.train = D.train[:config.debug_len]

        model = DocumentClassifier(config)
        model.to(self.device)
        B.to(self.device)

        steps = int(len(D.train)*config.c_num/config.bsz)
        optimizer = select_optimizer(model, config.optim, config.lr)
        self.loss_function = select_loss(config.loss)
        
        if config.scheduling: 
            triang = triangle_scheduler(optimizer, steps*config.epochs)
            scheduler = LambdaLR(optimizer, lr_lambda=triang)
        
        best_acc = -1
        print(f'BEGINNING TRAINING: ~{steps} BATCHES PER EPOCH')
        for epoch in range(config.epochs):
            #Training
            model.train()
            results = np.zeros(3)
            for k, batch in enumerate(B.batches(D.train, config.c_num, config.hier)):
                if config.hier:  loss, batch_acc = self.calc_batch_hier(model, batch)
                else:            loss, batch_acc = self.calc_batch(model, batch)
                results += [loss.item(), *batch_acc]
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if config.scheduling: scheduler.step()

                if k%config.print_len==0 and k!=0:
                    L.log(f'{epoch:<2} {k:<6} {results[0]/config.print_len:.3f}   {results[1]/results[2]:.4f}')
                    results = np.zeros(3)

            #Dev
            results = np.zeros(3)
            for k, batch in enumerate(B.batches(D.dev, config.c_num, config.hier)):
                if config.hier:  loss, batch_acc = self.calc_batch_hier(model, batch, no_grad=True)
                else:            loss, batch_acc = self.calc_batch(model, batch, no_grad=True)
                results += [loss.item(), *batch_acc]
            
            dev_loss, dev_acc = results[0]/k, results[1]/results[2]
            
            L.log(f'\n DEV  {dev_loss:.3f}   {dev_acc:.4f}\n')

            #Save model
            if config.save:
                if dev_acc > best_acc:
                    print(f'SAVING MODEL AT EPOCH {epoch}')
                    L.save_model(model)
                    best_acc = dev_acc 
               
            #GCDC correlation
            self.eval_GCDC(config)
            
            L.log(50*'--')

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

        if config.loss == 'ranking':
            self.loss_function = self.log_sigmoid_loss.ranking
        elif config.loss == 'classification':
            self.loss_function = self.log_sigmoid_loss.classification

        logger = np.zeros(3)
        with torch.no_grad():
            for k, batch in enumerate(tqdm(B.batches(D.test, c_num=20, hier=config.hier))):
                if config.hier:  loss, batch_acc = self.calc_batch_hier(model, batch, no_grad=True)
                else:            loss, batch_acc = self.calc_batch(model, batch, no_grad=True)
                logger += [loss.item(), *batch_acc]

        print('FINAL EVAL')
        print(f'\n{len(D.test):<5} {logger[0]/k:.3f}   {logger[1]/logger[2]:.4f}\n')
        
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
        print(f'GCDC correlation: PEAR {pearson:.3f}    SPEAR P {spearman:.3f}')

        
        
