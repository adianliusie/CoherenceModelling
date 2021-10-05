import json
import copy
import time
import numpy as np

class DataHandler():
    def __init__(self, data_src):
        if data_src in ['wsj', 'wiki']:
            self.train, self.dev, self.test = self.load_sents(data_src)
    
    def load_sents(self, data_src):
        paths = self.get_path(data_src)
        data = [self.load_data(path) for path in paths]
        data = [self.get_sents(dataset) for dataset in data]
        return data
    
    def load_data(self, path):
        with open(path) as jsonFile:
            data = json.load(jsonFile)
        return data
    
    def get_sents(self, data):
        return [i['sents'] for i in data]

    def get_path(self, data_src):
        if data_src == 'wiki': paths = self.wiki_paths()
        if data_src == 'wsj':  paths = self.wsj_paths()
        return paths
    
    def wiki_paths(self):
        base_dir = '/home/alta/Conversational/OET/al826/2021/data/unlabeled'
        paths = [f'{base_dir}/wiki_{i}.json' for i in ['train', 'dev', 'test']]
        return paths
    
    def wsj_paths(self):
        base_dir = '/home/alta/Conversational/OET/al826/2021/data/coherence'
        paths = [f'{base_dir}/WSJ_{i}.json' for i in ['train', 'dev', 'test']]
        return paths

    
class DataHandlerOLD():
    def __init__(self, data_src):
        path = self.get_path(data_src)
        data = self.load_data(path)
        self.data = [i['sents'] for i in data]
    
    def get_path(self, data_src):
        base_dir = '/home/alta/Conversational/OET/al826/2021'
        path_dict = {'wiki':f'{base_dir}/data/unlabeled/wiki_100000.json',
                     'WSJ_train':f'{base_dir}/data/coherence/WSJ_train.json',
                     'WSJ_test':f'{base_dir}/data/coherence/WSJ_test.json'}
        return path_dict[data_src]
    
    def load_data(self, path):
        with open(path) as jsonFile:
            return json.load(jsonFile)


