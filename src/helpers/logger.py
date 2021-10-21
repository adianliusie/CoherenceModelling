import os
import json
import torch

class Logger:
    def __init__(self, model_cfg, ptrain_cfg, train_cfg, exp_name=None):
        if model_cfg.save:
            self.exp_name = exp_name
            self.dir = self.get_dir()
            self.make_dir()
        else:
            self.dir = self.get_temp_dir()
            
        self.save_configs([model_cfg, ptrain_cfg, train_cfg])
        self.log = self.make_logger()
        self.record = self.make_result_logger()
                
    def get_temp_dir(self):
        base_dir = '/home/alta/Conversational/OET/al826/2021/coherence/results/temp'
        for file_ in os.listdir(base_dir):
            if os.path.isfile(f'{base_dir}/{file_}'): os.remove(f'{base_dir}/{file_}')
        for file_ in os.listdir(f'{base_dir}/models'):
            if os.path.isfile(f'{base_dir}/models/{file_}'): os.remove(f'{base_dir}/models/{file_}')
        return base_dir
        
    def get_dir(self):
        base_dir = f'/home/alta/Conversational/OET/al826/2021/coherence/results/synthetic_swp1/{self.exp_name}'
        return base_dir
       
    def make_dir(self):
        os.makedirs(self.dir)
        os.mkdir(f'{self.dir}/models')

    def save_configs(self, configs):
        cfg_names = ['model_config', 'ptrain_config', 'train_config']
        configs = {name:cfg.__dict__ for name, cfg in zip(cfg_names, configs) if cfg}
        config_path = f'{self.dir}/configs.json'
        with open(config_path, 'x') as jsonFile:
            json.dump(configs, jsonFile, indent=4)
    
    def make_result_logger(self):
        log_path = f'{self.dir}/results.txt' 
        open(log_path, 'w+').close()                  
        def log(*x):
            self.log(*x)
            with open(log_path, 'a') as f:
                for i in x:
                    f.write(str(i) + ' ')
                f.write('\n')
        return log
         
    def make_logger(self):
        log_path = f'{self.dir}/log.txt' 
        open(log_path, 'w+').close()                  
        def log(*x):
            print(*x)
            with open(log_path, 'a') as f:
                for i in x:
                    f.write(str(i) + ' ')
                f.write('\n')
        return log
    
    def save_model(self, name, model):
        device = next(model.parameters()).device
        model.to("cpu")
        torch.save(model.state_dict(), f'{self.dir}/models/{name}.pt')
        model.to(device)
    
    def monitor(self, value, mode):
        if not (hasattr(self, 'trace')):
            self.trace = {'train_loss':[], 'train_acc':[], 'dev_loss':[], 
                          'dev_acc':[], 'gcdc_spear':[], 'gcdc_acc':[], 
                          'gcdc_mean':[], 'gcdc_var':[], 'gcdc_mse':[]}

        if mode == 'train':
            self.trace['train_loss'].append(value[0])
            self.trace['train_acc'].append(value[1])
                
        elif mode == 'dev':
            self.trace['dev_loss'].append(value[0])
            self.trace['dev_acc'].append(value[1])
                
        elif mode == 'gcdc':
            for key in vars(value):
                self.trace[f'gcdc_{key}'].append(getattr(value, key))
        
        self.update_loss_curves()
                
    def update_loss_curves(self):
        with open(f'{self.dir}/train_curves.json', 'w+') as jsonFile:
            json.dump(self.trace, jsonFile)
    
    @property
    def path(self):
        return self.dir