from torch.optim.lr_scheduler import LambdaLR
import torch
import torch.nn as nn
import numpy as np
from scipy import stats
import random 
from tqdm import tqdm
from types import SimpleNamespace
from .utils import Batcher, DataHandler, Logger
from .utils.config import select_optimizer, select_loss
from .models import DocumentClassifier

def toggle_grad(func):
    def inner(*args, no_grad=False):
        if no_grad==True:
            with torch.no_grad():
                return func(*args)
        else:
            return func(*args)
    return inner

class ExperimentHandler:
    def __init__(self, model_cfg, ptrain_cfg=None, train_cfg=None):
        
        self.device = torch.device('cuda:0') if torch.cuda.is_available() \
                      else torch.device('cpu')
        if model_cfg.device == 'cpu': self.device = torch.device('cpu')
            
        self.model_cfg  = model_cfg
        self.ptrain_cfg = ptrain_cfg
        self.train_cfg  = train_cfg
        
        self.L = Logger(model_cfg, ptrain_cfg, train_cfg)
        self.model = DocumentClassifier(model_cfg)
        
        if ptrain_cfg: self.pair_loss_fn = select_loss(ptrain_cfg.loss)
        if train_cfg:  self.sup_loss_fn  = select_loss(train_cfg.loss)
    
        self.hier = model_cfg.hier
        self.system = model_cfg.system
        self.run()
        
    def run(self):
        if self.ptrain_cfg is not None:
            self.corrupted_pre_training(self.ptrain_cfg)
            self.load_pretrain()
            
        if self.train_cfg is not None:
            if self.train_cfg.loss == 'mse':
                self.regression_training(self.train_cfg)
            elif self.train_cfg.loss == 'cross_loss':
                self.classification_training(self.train_cfg)

    def corrupted_pre_training(self, config):
        self.L.log(f'Corrupted Training Parameters: \n {config}')
        
        #######     Set up     #######
        D = DataHandler(config.data_src)
        B = Batcher(self.system, self.hier, config.bsz, 
                    config.max_len, config.schemes, config.args) 
        
        if config.debug_len: 
            D.train = D.train[:config.debug_len]
        
        self.model.to(self.device)
        B.to(self.device)
        
        steps = int(len(D.train)*config.c_num/config.bsz)
        optimizer = select_optimizer(self.model, config.optim, config.lr)
        
        if config.scheduling: 
            triang = triangle_scheduler(optimizer, steps*config.epochs)
            scheduler = LambdaLR(optimizer, lr_lambda=triang)
        
        best_metric = -1
        print(f'BEGINNING TRAINING: ~{steps} BATCHES PER EPOCH')
        for epoch in range(1, config.epochs+1):
            #######     Training     #######
            self.model.train()
            results = np.zeros(3)
            for k, batch in enumerate(B.batches(D.train, config.c_num)):
                if self.hier:   b_out = self.pair_loss_hier(batch)
                else:           b_out = self.pair_loss(batch)
                results += [b_out.loss.item(), *b_out.acc]
                
                optimizer.zero_grad()
                b_out.loss.backward()
                optimizer.step()
                if config.scheduling: scheduler.step()
                
                if k%config.print_len==0 and k!=0:
                    self.L.log(f'{epoch:<2} {k:<6} {results[0]/config.print_len:.3f}'\
                    f'    {results[1]/results[2]:.4f}')
                    results = np.zeros(3)

                #######    Save model    #######
                if k%config.check_len==0 and k!=0:                    
                    metric = self.gcdc_evaluation(config, printing=True)

                    if metric > best_metric:
                        self.L.log(f'SAVING MODEL AT EPOCH {epoch}')
                        self.L.save_model('pre_train', self.model)
                        best_metric = metric 

            #######       Dev        #######
            results = np.zeros(3)
            for k, batch in enumerate(B.batches(D.dev, config.c_num)):
                if self.hier:   b_out = self.pair_loss_hier(batch, no_grad=True)
                else:           b_out = self.pair_loss(batch, no_grad=True)
                results += [b_out.loss.item(), *b_out.acc]
            dev_loss, dev_acc = results[0]/k, results[1]/results[2]
            self.L.log(f'\n DEV  {dev_loss:.3f}   {dev_acc:.4f}\n')

    def regression_training(self, config):
        self.L.log(f'Supervised Training Parameters: \n {config}')

        #######     Set up     ####### 
        D = DataHandler('gcdc')
        B = Batcher(self.system, self.hier, config.bsz, config.max_len) 
        self.model.to(self.device)
        B.to(self.device)

        steps = int(len(D.train)/config.bsz)
        optimizer = select_optimizer(self.model, config.optim, config.lr)
        
        if config.scheduling: 
            triang = triangle_scheduler(optimizer, steps*config.epochs)
            scheduler = LambdaLR(optimizer, lr_lambda=triang)

        #train_sets = [D.clinton_train, D.enron_train, D.yahoo_train, D.yelp_train]
        #if config.lim: train_sets = [data_set[:config.data_lim] for data_set in train_sets]
        
        #for data_set in train_sets:
        data_set = D.clinton_train[:config.data_lim]
        for epoch in range(1, config.epochs+1):
            results = np.zeros(3)
            for k, batch in enumerate(B.labelled_batches(data_set)):
                if self.hier:   b_out = self.sup_loss_hier(batch)
                else:           b_out = self.sup_loss(batch)
                hits = self.class_perf(b_out.pred, b_out.labels)
                results += [b_out.loss.item(), hits, len(b_out.labels)]
                
                optimizer.zero_grad()
                b_out.loss.backward()
                optimizer.step()
                if config.scheduling: scheduler.step()

                if k%config.print_len==0 and k!=0:
                    self.L.log(f'{epoch:<2} {k:<6} {results[0]/config.print_len:.3f}'\
                    f'    {results[1]/results[2]:.4f}')
                    results = np.zeros(3)
                    
            self.gcdc_evaluation(config)
          
    def classification_training(self, config):
        self.L.log(f'Supervised Training Parameters: \n {config}')

        #######     Set up     #######
        D = DataHandler('gcdc')
        B = Batcher(self.system, self.hier, config.bsz, config.max_len) 
        self.model.classifier = nn.Linear(300, 3)            
        self.model.to(self.device)
        B.to(self.device)

        steps = int(len(D.train)/config.bsz)
        optimizer = select_optimizer(self.model, config.optim, config.lr)
        
        if config.scheduling: 
            triang = triangle_scheduler(optimizer, steps*config.epochs)
            scheduler = LambdaLR(optimizer, lr_lambda=triang)

        data_set = D.clinton_train[:config.data_lim]
        for epoch in range(1, config.epochs+1):
            results = np.zeros(3)
            for k, batch in enumerate(B.labelled_batches(data_set, classify=True)):
                if self.hier:   b_out = self.sup_loss_hier(batch)
                else:           b_out = self.sup_loss(batch)    
                hits = sum([pred == lab for pred, lab in zip(b_out.pred, b_out.labels)])
                results += [b_out.loss.item(), hits, len(b_out.labels)]

                optimizer.zero_grad()
                b_out.loss.backward()
                optimizer.step()
                if config.scheduling: scheduler.step()

                if k%config.print_len==0 and k!=0:
                    self.L.log(f'{epoch:<2} {k:<6} {results[0]/config.print_len:.3f}'\
                    f'    {results[1]/results[2]:.4f}')
                    results = np.zeros(3)
            
            self.gcdc_evaluation(config, classify=True)
      
    def corrupt_eval(self, config=None):
        D = DataHandler(config.data_src)
        B = Batcher(config.system, config.bsz, config.max_len,
                    config.schemes, config.args) 
        
        self.model.eval()
        B.to(self.device)

        random.seed(10)
        logger = np.zeros(3)
        with torch.no_grad():
            for k, batch in enumerate(B.batches(D.test, c_num=5, hier=config.hier)):
                if self.hier:   b_output = self.pair_loss_hier(batch, no_grad=True)
                else:           b_output = self.pair_loss(batch, no_grad=True)
                results += [b_output.loss.item(), *batch_output.acc]

        self.L.log('FINAL EVAL')
        self.L.log(f'\n{len(D.test):<5} {logger[0]/k:.3f}   {logger[1]/logger[2]:.4f}\n')
        
    def gcdc_evaluation(self, config, printing=True, mode='dev', classify=False):
        D = DataHandler('gcdc')
        B = Batcher(self.system, self.hier, config.bsz, config.max_len) 
        self.model.to(self.device)
        B.to(self.device)
        
        eval_sets = [D.clinton_dev,  D.enron_dev,  D.yahoo_dev,  D.yelp_dev ]
        if mode == 'test':  
            eval_sets = [D.clinton_test, D.enron_test, D.yahoo_test, D.yelp_test]

        performance = []
        for data_set in eval_sets:
            predictions, scores  = [], []
            for k, batch in enumerate(B.labelled_batches(data_set, classify=classify)):
                if self.hier:   b_out = self.sup_loss_hier(batch, no_grad=True)
                else:           b_out = self.sup_loss(batch, no_grad=True)
                predictions += b_out.pred
                scores      += b_out.labels
                
            if classify:
                hits = sum([pred == lab for pred, lab in zip(predictions, scores)])
                performance.append(hits/len(predictions))
                self.L.log(hits/len(predictions))
                metric = hits
            else:  
                MSE, spearman = self.reg_perf(predictions, scores)
                acc = self.class_perf(predictions, scores)/len(predictions)
                performance.append([MSE, spearman, acc])
                metric = sum([perf[1] for perf in performance])/len(performance)
        if printing: self.print_gcdc_perf(performance)
        return metric
    
    def reg_perf(self, predictions, scores):
        predictions, scores = np.array(predictions), np.array(scores)
        pearson = stats.pearsonr(predictions, scores)[0]
        spearman = stats.spearmanr(predictions, scores)[0]
        MSE = np.mean((predictions-scores)**2)
        return MSE, spearman
    
    def class_perf(self, predictions, scores): 
        count = 0
        for pred, score in zip(predictions, scores):
            if self._round(pred) == self._round(score): count +=1
        return count
    
    def _round(self, pred):
        if   pred<5.4: output=1
        elif pred<6.6: output=2
        else:          output=3
        return output
    
    def print_gcdc_perf(self, performance):
        domains = ['clinton', 'enron', 'yahoo', 'yelp']
        self.L.log('--'*40)
        for name, perf in zip(domains, performance):
            self.L.log(f'{name:<10}  MSE:{perf[0]:5.2f}  '\
            f'spear:{perf[1]:4.3f}  acc:{perf[2]:.2f}')
        self.L.log('--'*40)

    @toggle_grad
    def pair_loss(self, batch):
        pos, neg = batch
        y_pos = self.model(pos.ids, pos.mask)
        y_neg = self.model(neg.ids, neg.mask)
        loss = self.pair_loss_fn(y_pos, y_neg)
        acc = [sum(y_pos - y_neg > 0).item(), len(y_pos)]
        return_dict = {'loss':loss, 'acc':acc}
        return SimpleNamespace(**return_dict)

    @toggle_grad
    def pair_loss_hier(self, batch):
        loss, acc = 0, np.zeros(2)
        for pos, neg in batch:
            y_pos = self.model(pos.ids, pos.mask)
            y_neg = self.model(neg.ids, neg.mask)
            loss += self.pair_loss_fn(y_pos, y_neg)/len(batch)
            acc += [(y_pos>y_neg).item(), 1]
        return_dict = {'loss':loss, 'acc':acc}
        return SimpleNamespace(**return_dict)

    @toggle_grad
    def sup_loss(self, batch):
        y = self.model(batch.ids, batch.mask)
        loss = self.sup_loss_fn(y, batch.score)
        if len(y.shape) == 2:
            y = torch.argmax(y, -1)
        return_dict = {'loss':loss, 'pred':y.tolist(), 
                       'labels':batch.score.tolist()}
        return SimpleNamespace(**return_dict)

    @toggle_grad
    def sup_loss_hier(self, batch):
        loss, preds, labels = 0, [], []
        for doc in batch:
            y = self.model(doc.ids, doc.mask)
            preds.append(y.item())
            labels.append(doc.score.item())
            loss += self.sup_loss_fn(y, doc.score)/len(batch)
        return_dict = {'loss':loss, 'pred':preds, 'labels':labels}
        return SimpleNamespace(**return_dict)
    
    def load_pretrain(self):
        self.model.load_state_dict(torch.load(self.L.dir + '/pre_train.pt'))
        self.model.to(self.device)
