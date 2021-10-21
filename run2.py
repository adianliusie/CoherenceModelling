from src.experiment import ExperimentHandler
from types import SimpleNamespace

from types import SimpleNamespace

model_cfg  = {'system':'glove', 'pooling':'attention', 'hier':True, 'device':'cuda:0',
              'save':True, 'exp_name':'glove_rank_long', 'ensemble':None}

ptrain_cfg = {'bsz':4, 'lr':2e-4, 'epochs':1, 'scheduling':False, 'optim':'adam', 
              'data_src':'wiki', 'loss':'ranking', 'c_num':1, 'schemes':[2], 'args':[1],
              'max_len':512, 'print_len':100, 'data_lim':500_000, 'check_len':None,
              'reg':None, 'reg_lr':0.001}

train_cfg  = {'bsz':4, 'lr':1e-5, 'epochs':8, 'scheduling':False, 'optim':'adam',
              'data_src':'gcdc', 'data_lim':None, 'loss':'mse', 'max_len':512, 'print_len':40}

if model_cfg['system'] == 'bert':
    assert ptrain_cfg['lr'] < 5e-4

if model_cfg['system'] == 'glove':
    assert ptrain_cfg['lr'] > 5e-5

model_cfg  = SimpleNamespace(**model_cfg)
ptrain_cfg = SimpleNamespace(**ptrain_cfg)
train_cfg  = SimpleNamespace(**train_cfg)

#ptrain_cfg=None
E = ExperimentHandler(model_cfg, ptrain_cfg, train_cfg)
E.run()
