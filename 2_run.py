from src.experiment import ExperimentHandler
from types import SimpleNamespace

model_cfg  = {'system':'roberta', 'pooling':'first', 'hier':False, 'device':'cuda:0',
              'save':True, 'exp_name':'clinton_pre_l2', 'ensemble':5}

train_cfg  = {'bsz':4, 'lr':1e-5, 'epochs':8, 'scheduling':False, 'optim':'adam',
              'data_src':'clinton', 'data_lim':None, 'loss':'mse', 'max_len':512, 'print_len':40}

model_cfg  = SimpleNamespace(**model_cfg)
train_cfg  = SimpleNamespace(**train_cfg)

E = ExperimentHandler(model_cfg, None, train_cfg)
E.finetune_experiment()
