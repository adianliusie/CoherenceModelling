import os
import json
import torch

class Logger:
    def __init__(self, model_cfg, ptrain_cfg, train_cfg):
        if model_cfg.save:
            self.dir = self.get_dir(model_cfg.system, model_cfg.hier)
            self.make_dir()
        else:
            self.dir = self.get_temp_dir()
            
        self.save_configs([model_cfg, ptrain_cfg, train_cfg])
        self.log = self.make_logger()
    
    def get_temp_dir(self):
        base_dir = '/home/alta/Conversational/OET/al826/2021/coherence/results/temp'
        for file_ in os.listdir(base_dir):
            os.remove(f'{base_dir}/{file_}')
        return base_dir
        
    def get_dir(self, *POI):
        current_exp = 'finetuning-batch'
        root_dir = f'/home/alta/Conversational/OET/al826/2021/coherence/results/{current_exp}'
        exp_name = '_'.join([str(i) for i in POI])
        base_dir = f'{root_dir}/{exp_name}'
        return base_dir
       
    def make_dir(self):
        if os.path.isdir(self.dir):
            self.dir += '2'
            os.mkdir(self.dir)
        else:
            os.makedirs(self.dir)

    def save_configs(self, configs):
        cfg_names = ['model_config', 'ptrain_config', 'train_config']
        configs = {name:cfg.__dict__ for name, cfg in zip(cfg_names, configs) if cfg}
        config_path = f'{self.dir}/configs.json'
        with open(config_path, 'x') as jsonFile:
            json.dump(configs, jsonFile, indent=4)
            
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
        torch.save(model.state_dict(), f'{self.dir}/{name}.pt')
        model.to(device)
    
    @property
    def path(self):
        return self.dir