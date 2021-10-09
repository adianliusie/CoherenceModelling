import os
import json
import torch

class Logger:
    def __init__(self, config):
        if config.save:
            self.dir = self.get_dir(config)
            self.make_dir(config)
            self.save_config(config)
            self.log = self.make_logger()
        else:
            self.log = print
            
    def get_dir(self, config):
        root_dir = '/home/alta/Conversational/OET/al826/2021/coherence/results/'
        system = f'{config.system}-hier' if config.hier else config.system
        name = f'{system}-{config.loss}-{config.bsz}b'
        base_dir = f'{root_dir}/{config.data_src}/{name}'
        return base_dir
       
    def make_dir(self, config):
        if os.path.isdir(self.dir):
            self.dir += '2'
            os.mkdir(self.dir)
        else:
            os.makedirs(self.dir)

    def save_config(self, config):
        config_path = f'{self.dir}/config.json'
        with open(config_path, 'x') as jsonFile:
            json.dump(config._asdict(), jsonFile)

    def make_logger(self):
        log_path = f'{self.dir}/log.txt' 
        open(log_path, 'w+').close()                  
        def log(*x):
            print(*x)
            with open(log_path, 'a') as f:
                for i in x:
                    f.write(i)
                f.write('\n')
        return log
    
    def save_model(self, model):
        device = next(model.parameters()).device
        model.to("cpu")
        torch.save(model.state_dict(), f'{self.dir}/model.pt')
        model.to(device)
        