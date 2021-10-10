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
    def __init__(self, config):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() \
                      else torch.device('cpu')
        
        self.config = config
        self.L = Logger(config)
        self.model = DocumentClassifier(config)
        
        self.pair_loss_fn = select_loss(config.loss)

    def corrupted_training(self, config=None):
        if config == None: config = self.config
        self.L.log(f'Parameters: {config}')
        
        # First part of the script is set up
        D = DataHandler(config.data_src)
        B = Batcher(config.system, config.bsz, config.max_len,
                    config.schemes, config.args) 
        
        if config.debug_len: 
            D.train = D.train[:config.debug_len]

        self.model.to(self.device)
        B.to(self.device)

        steps = int(len(D.train)*config.c_num/config.bsz)
        optimizer = select_optimizer(self.model, config.optim, config.lr)
        
        if config.scheduling: 
            triang = triangle_scheduler(optimizer, steps*config.epochs)
            scheduler = LambdaLR(optimizer, lr_lambda=triang)
        
        best_acc = -1
        print(f'BEGINNING TRAINING: ~{steps} BATCHES PER EPOCH')
        for epoch in range(config.epochs):
            #Training
            self.model.train()
            results = np.zeros(3)
            for k, batch in enumerate(B.batches(D.train, config.c_num, config.hier)):
                if config.hier:  loss, batch_acc = self.pair_loss_hier(batch)
                else:            loss, batch_acc = self.pair_loss(batch)
                results += [loss.item(), *batch_acc]
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if config.scheduling: scheduler.step()

                if k%config.print_len==0 and k!=0:
                    self.L.log(f'{epoch:<2} {k:<6} {results[0]/config.print_len:.3f}   {results[1]/results[2]:.4f}')
                    results = np.zeros(3)

            #Dev
            results = np.zeros(3)
            for k, batch in enumerate(B.batches(D.dev, config.c_num, config.hier)):
                if config.hier:  loss, batch_acc = self.pair_loss_hier(batch, no_grad=True)
                else:            loss, batch_acc = self.pair_loss(batch, no_grad=True)
                results += [loss.item(), *batch_acc]
            
            dev_loss, dev_acc = results[0]/k, results[1]/results[2]
            
            self.L.log(f'\n DEV  {dev_loss:.3f}   {dev_acc:.4f}\n')

            #Save model
            if dev_acc > best_acc:
                print(f'SAVING MODEL AT EPOCH {epoch}')
                self.L.save_model(self.model)
                best_acc = dev_acc 

            #GCDC correlation
            spearman = self.GCDC_evaluation(config)
            self.L.log('GCDC correlation', spearman)
            self.L.log(50*'--')

    def corrupt_eval(self, config=None):
        D = DataHandler(config.data_src)
        B = Batcher(config.system, config.bsz, config.max_len,
                    config.schemes, config.args) 
        
        self.model.load_state_dict(torch.load(self.L.model_path))
        self.model.to(self.device)
        self.model.eval()
        B.to(self.device)

        random.seed(10)
        logger = np.zeros(3)
        with torch.no_grad():
            for k, batch in enumerate(tqdm(B.batches(D.test, c_num=5, hier=config.hier))):
                if config.hier:  loss, batch_acc = self.pair_loss_hier(batch, no_grad=True)
                else:            loss, batch_acc = self.pair_loss(batch, no_grad=True)
                logger += [loss.item(), *batch_acc]

        self.L.log('FINAL EVAL')
        self.L.log(f'\n{len(D.test):<5} {logger[0]/k:.3f}   {logger[1]/logger[2]:.4f}\n')
        
    def GCDC_evaluation(self, mode='dev', config=None):
        if config == None: config = self.config

        D = DataHandler('gcdc')
        B = Batcher(config.system, config.bsz, config.max_len) 
        self.model.to(self.device)
        B.to(self.device)

        if mode == 'dev':  eval_sets = [D.clinton_dev,  D.enron_dev,  D.yahoo_dev,  D.yelp_dev]
        else:            eval_sets = [D.clinton_test, D.enron_test, D.yahoo_test, D.yelp_test]

        for data_set in eval_sets:
            scores = []
            predictions = []

            for k, batch in enumerate(B.labelled_batches(data_set, config.hier)):
                if config.hier:
                    for doc in batch:
                        y = self.model(doc.ids, doc.mask)
                        predictions.append(y.item())
                        scores.append(doc.score.item())
                else:
                    y = self.model(batch.ids, batch.mask)
                    predictions += y.tolist()
                    scores += batch.score.tolist()
             
            MSE, spearman = self.reg_perf(predictions, scores)
            print(MSE, spearman)
        return spearman
    
    def reg_perf(self, predictions, scores):
        predictions, scores = np.array(predictions), np.array(scores)
        pearson = stats.pearsonr(predictions, scores)[0]
        spearman = stats.spearmanr(predictions, scores)[0]
        MSE = np.mean((predictions-scores)**2)
        return MSE, spearman

    def pair_loss(self, batch, no_grad=False):
        if no_grad==True:
            with torch.no_grad():
                return self.pair_loss(batch, no_grad=False)

        pos, neg = batch
        y_pos = self.model(pos.ids, pos.mask)
        y_neg = self.model(neg.ids, neg.mask)

        loss = self.pair_loss_fn(y_pos, y_neg)
        acc = [sum(y_pos - y_neg > 0).item(), len(y_pos)]
        return loss, acc

    def pair_loss_hier(self, model, batch, no_grad=False):
        if no_grad==True:
            with torch.no_grad():
                return self.pair_loss_hier(batch, no_grad=False)
            
        loss, acc = 0, np.zeros(2)
        for pos, neg in batch:
            y_pos = self.model(pos.ids, pos.mask)
            y_neg = self.model(neg.ids, neg.mask)
            loss += self.pair_loss_fn(y_pos, y_neg)/len(batch)
            acc += [(y_pos>y_neg).item(), 1]
        return loss, acc

        
        
