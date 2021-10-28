from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import torch
from types import SimpleNamespace
import numpy as np
from scipy import stats
import random 
from tqdm import tqdm
import copy 
from .models import DocumentClassifier
from .helpers import Batcher, DataHandler, Logger
from .utils import select_optimizer, select_loss, gaussian_pdf, triangle_scheduler, toggle_grad

class ExperimentHandler:
    def __init__(self, model_cfg, ptrain_cfg=None, train_cfg=None):
        
        self.device = torch.device(model_cfg.device) if torch.cuda.is_available() \
                      else torch.device('cpu')
        
        self.model_cfg  = model_cfg
        self.ptrain_cfg = ptrain_cfg
        self.train_cfg  = train_cfg
        
        self.L = Logger(model_cfg, ptrain_cfg, train_cfg, model_cfg.exp_name)
        self.model = DocumentClassifier(model_cfg)
        
        self.L.log(f'Model Parameters: \n {model_cfg} \n')

        if ptrain_cfg: self.pair_loss_fn = select_loss(ptrain_cfg.loss)
        if train_cfg:  self.sup_loss_fn  = select_loss(train_cfg.loss)
    
        self.hier = model_cfg.hier
        self.system = model_cfg.system
        
    def pretrain_experiment(self):
        self.corrupted_pre_training(self.ptrain_cfg)
        self.load_pretrain()
    
    def finetune_experiment(self):
        cfg = copy.copy(self.train_cfg)
 
        self.ensemble = [[],[],[],[]]
        self.test_labels = [None,None,None,None]

        for r in range(self.model_cfg.ensemble):
            self.model = DocumentClassifier(self.model_cfg)
            #path = '/home/alta/Conversational/OET/al826/2021/coherence/results/finetuning/rob_shf_base/models/pre_train.pt'       
            #self.model.load_state_dict(torch.load(path))
            # if self.ptrain_cfg: self.load_model(name='pre_train')
            self.regression_training(cfg)

        mean_result, sigmas = self.get_mean_performance(self.ensemble)
        self.print_gcdc(mean_result, record=True, prefix=f'mean')
        self.print_gcdc(sigmas, record=True, prefix=f'std')

        ensemble_preds = np.mean(self.ensemble, axis=1)
        result = self.eval_gcdc_set(ensemble_preds, self.test_labels)
        self.print_gcdc(result, record=True, prefix=f'ensemble')
        
    def get_mean_performance(self, ensemble):      
        new_dict = lambda : {'mse':0, 'spear':0, 'mean':0, 'var':0, 'acc':0}
        
        output_means = []
        output_std = []

        for k, domain in enumerate(ensemble):
            perf_set = [self.eval_gcdc_preds(pred, self.test_labels[k]) for pred in domain]
            means, sigmas = new_dict(), new_dict()
            for key in means:
                values = np.array([getattr(perf, key) for perf in perf_set])
                means[key] = values.mean()
                sigmas[key] = values.std()
            output_means.append(SimpleNamespace(**means))
            output_std.append(SimpleNamespace(**sigmas))

        return output_means, output_std

    def corrupted_pre_training(self, config):        
        #######     Set up     #######
        D = DataHandler(config.data_src)
        B = Batcher(self.system, self.hier, config.bsz, 
                    config.max_len, config.schemes, config.args) 
        
        if config.data_lim: D.train = D.train[:config.data_lim]
        self.model.to(self.device), B.to(self.device)
 
        steps = int(len(D.train)*config.c_num/config.bsz)
        optimizer = select_optimizer(self.model, config.optim, config.lr)       
        if config.scheduling: 
            triang = triangle_scheduler(optimizer, steps*config.epochs)
            scheduler = LambdaLR(optimizer, lr_lambda=triang)
        
        if config.reg: 
            self.set_regularisation(config.reg, lr=config.reg_lr, mode='dev')
            
        best_metric = -1
        print(f'BEGINNING TRAINING: ~{steps} BATCHES PER EPOCH')
        for epoch in range(1, config.epochs+1):
            #######     Training     #######
            self.model.train()
            results = np.zeros(3)
            for k, batch in enumerate(B.batches(D.train, config.c_num), start=1):
                if self.hier:   b_out = self.pair_loss_hier(batch)
                else:           b_out = self.pair_loss(batch)
                    
                loss, acc = b_out.loss.item(), b_out.acc[0]/b_out.acc[1]
                results += [loss, *b_out.acc]
                self.L.monitor([loss, acc, b_out.pos_mean, b_out.neg_mean], mode='train')
                
                optimizer.zero_grad()
                b_out.loss.backward()
                optimizer.step()
                if config.scheduling: scheduler.step()
                
                if k%config.print_len==0 and k!=0:
                    loss, acc = results[0]/config.print_len, results[1]/results[2]
                    self.L.log(f'{epoch:<2}  {k:<6}  {loss:.3f}  {acc:.4f}')
                    results = np.zeros(3)
                
                #######    Save model    #######
                if k%config.check_len==0:   
                    gcdc_perf = self.gcdc_evaluation(config)
                    self.print_gcdc(gcdc_perf, prefix=f'ptrain {k}')
                    self.L.monitor(gcdc_perf, mode='gcdc')
          
                    metric = sum([x.spear for x in gcdc_perf])          
                    if metric > best_metric:
                        self.L.save_model('pre_train', self.model)
                        best_metric = metric
                        print(f'MODEL SAVED: dev acc {best_metric}')

        self.load_model(name='pre_train')
        performance = self.corrupt_eval(D, B, mode='test')
        self.L.log(performance)

        performance = self.gcdc_evaluation(config, mode='test')
        self.print_gcdc(performance, record=True, prefix=epoch)
        self.L.log(performance)
        self.L.log('')

    def regression_training(self, config):
        #######     Set up     ####### 
        D = DataHandler('gcdc')
        B = Batcher(self.system, self.hier, config.bsz, config.max_len) 
        self.model.to(self.device)
        B.to(self.device)
        
        if config.data_src in ['clinton', 'enron', 'yahoo', 'yelp']:
            train = getattr(D, f'{config.data_src}_train')
            ind = ['clinton', 'enron', 'yahoo', 'yelp'].index(config.data_src)
            
        elif config.data_src in ['all']:
            train = D.train
   
        train = train[:config.data_lim]
        
        optimizer = select_optimizer(self.model, config.optim, config.lr)
        steps = int(len(train)/config.bsz)
        if config.scheduling: 
            triang = triangle_scheduler(steps*config.epochs)
            scheduler = LambdaLR(optimizer, lr_lambda=triang)

        best_metric, best_epoch = 1000, 0
        for epoch in range(1, config.epochs+1):
            #######     Training     #######
            for k, batch in enumerate(B.labelled_batches(train)):
                if self.hier:   b_out = self.sup_loss_hier(batch)
                else:           b_out = self.sup_loss(batch)
                hits = self.gcdc_accuracy(b_out.pred, b_out.labels)
                
                optimizer.zero_grad()
                b_out.loss.backward()
                optimizer.step()
                if config.scheduling: scheduler.step()

            #######       Dev        #######
            perf = self.gcdc_evaluation(config)
            self.print_gcdc(perf, prefix=epoch, lim=ind)
            self.L.monitor(perf, mode='gcdc')

            if perf[ind].mse < best_metric:
                print(epoch, 'saving')
                best_epoch = epoch
                self.L.save_model(f'finetune', self.model)
                best_metric = perf[ind].mse
        
        self.load_model(name=f'finetune')
        result = self.gcdc_evaluation(config, mode='test')
        self.print_gcdc(result, record=True, prefix=f'TEST (e{best_epoch})')
        self.L.log('')

    @toggle_grad
    def pair_loss(self, batch):
        pos, neg = batch
        y_pos = self.model(pos.ids, pos.mask)
        y_neg = self.model(neg.ids, neg.mask)
        loss = self.pair_loss_fn(y_pos, y_neg)
        if hasattr(self, 'regularisation'):
            loss += self.regularisation(y_pos)
        acc = [sum(y_pos - y_neg > 0).item(), len(y_pos)]
        return_dict = {'loss':loss, 'acc':acc, 
                       'pos_mean':torch.mean(y_pos).item(),
                       'neg_mean':torch.mean(y_neg).item()}
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
            loss += self.sup_loss_fn(y, doc.score)/len(batch)
            if len(y.shape) == 2:
                y = torch.argmax(y, -1)
            preds.append(y.item())
            labels.append(doc.score.item())

        return_dict = {'loss':loss, 'pred':preds, 'labels':labels}
        return SimpleNamespace(**return_dict)

    def gcdc_evaluation(self, config, mode='dev'):
        D = DataHandler('gcdc')
        B = Batcher(self.system, self.hier, config.bsz, config.max_len) 
        self.model.to(self.device)
        B.to(self.device)
    
        if mode == 'dev':
            eval_set = [D.clinton_dev, D.enron_dev, D.yahoo_dev, D.yelp_dev]
        elif mode == 'test':
            eval_set = [D.clinton_test, D.enron_test, D.yahoo_test, D.yelp_test]

        output = []
        
        for k, data_set in enumerate(eval_set):
            predictions, labels  = [], []
            for batch in B.labelled_batches(data_set, shuffle=False):
                if self.hier:   b_out = self.sup_loss_hier(batch, no_grad=True)
                else:           b_out = self.sup_loss(batch, no_grad=True)
                predictions += b_out.pred
                labels += b_out.labels
        
            if mode=='test' and hasattr(self, 'ensemble'):
                self.ensemble[k].append(predictions)
                self.test_labels[k] = labels
            
            performance = self.eval_gcdc_preds(predictions, labels)
            output.append(performance)
        return output
    
    def eval_gcdc_preds(self, predictions, labels):
        predictions, labels = np.array(predictions), np.array(labels)
        pearson = stats.pearsonr(predictions, labels)[0]
        spearman = stats.spearmanr(predictions, labels)[0]
        MSE = np.mean((predictions-labels)**2)
        mean, variance = predictions.mean(), predictions.var()
        acc = self.gcdc_accuracy(predictions, labels)
        output = {'mse':MSE, 'spear':spearman, 'mean':mean, 'var':variance, 'acc':acc}
        output = SimpleNamespace(**output)
        return output
    
    def eval_gcdc_set(self, predictions, labels):
        output = []
        for pred, lab in zip(predictions, labels):
            perf = self.eval_gcdc_preds(pred, lab)
            output.append(perf)
        return output
                      
    def gcdc_accuracy(self, predictions, scores): 
        def rnd(pred):
            if   pred<5.4: output=1
            elif pred<6.6: output=2
            else:          output=3
            return output

        count = 0
        for pred, score in zip(predictions, scores):
            if rnd(pred) == rnd(score): count +=1
        return count/len(predictions)

    def print_gcdc(self, x_set, record=False, prefix='', lim=-1):
        strings = []
        domains = ['clinton', 'enron', 'yahoo', 'yelp']
        for name, x in zip(domains, x_set):
            string = f'{prefix} {name:<8}  MSE:{x.mse:.2f}  spear:{x.spear:.3f}  '\
                     f'acc:{x.acc:.2f}  mean:{x.mean:.3f}  var:{x.var:.3f}'
            strings.append(string)

        if record: self.L.record('')
        else: self.L.log('')
            
        if 0<=lim<=3: strings = [strings[lim]]
        for string in strings:
            if record: self.L.record(string)
            else: self.L.log(string)

            
    def corrupt_eval(self, D, B, mode='test'):        
        if   mode == 'dev' : dataset = D.dev[:1000]
        elif mode == 'test': dataset = D.test
        
        random.seed(10)
        results = np.zeros(3)
        with torch.no_grad():
            for k, batch in tqdm(enumerate(B.batches(dataset, c_num=5), start=1)):
                if self.hier:   b_output = self.pair_loss_hier(batch, no_grad=True)
                else:           b_output = self.pair_loss(batch, no_grad=True)
                results += [b_output.loss.item(), *b_output.acc]
        return (results[0]/k, results[1]/results[2])
           
    def set_regularisation(self, reg='l2', lr=0.1, mode='dev'):
        D = DataHandler('gcdc')
        data_set = D.clinton_dev
        if mode == 'train':  
            eval_set = D.clinton_train
 
        scores = np.array([ex.score for ex in data_set])
        mean, variance = scores.mean(), scores.var()
        self.L.log(f'mean: {mean:.3f}  variance: {variance:.3f}')

        G = gaussian_pdf(mean, variance)
        def gaussian_loss(x):
            probs = G(x)
            loss = -1*lr*torch.mean(torch.log(probs))
            return loss
         
        def L2_loss(x):
            return lr*torch.mean((x-mean)**2)

        if reg == 'gaussian': self.regularisation = gaussian_loss
        if reg == 'l2':       self.regularisation = L2_loss
            
        if hasattr(self, 'regularisation'):
            print('regularisation set up')
 
    def load_model(self, name='pre_train'):
        self.model.load_state_dict(torch.load(self.L.dir + f'/models/{name}.pt'))
        self.model.to(self.device)
