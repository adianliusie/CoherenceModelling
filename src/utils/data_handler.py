from types import SimpleNamespace
import json
import copy
import time
import numpy as np

class DataHandler():
    def __init__(self, data_src):
        if data_src in ['wsj', 'wiki', 'wiki_small']:
            self.train, self.dev, self.test = self.parse_data(data_src)
 
        if data_src in ['wiki_tiny']:
            self.train, self.dev, self.test = self.parse_data('wiki_small')
            self.train = self.train[:10_000]
            self.dev = self.dev[:2_000]
            self.test = self.test[:2_000]

        if data_src in ['gcdc']:
            domains, sets = ['clinton', 'enron', 'yelp', 'yahoo'], ['train', 'test']
            data_name = [f'{domain}_{set1}' for domain in domains for set1 in sets]
            data_sets = self.parse_data(data_src)
            
            for name, data in zip(data_name, data_sets):
                setattr(self, name, data)
            
            self.train = self.clinton_train + self.enron_train + self.yelp_train + self.yahoo_train
            self.test = self.clinton_test + self.enron_test + self.yelp_test + self.yahoo_test
            self.dev = self.test
            
    def parse_data(self, data_src):
        paths = self.get_path(data_src)
        data = [self.load_data(path) for path in paths]
        data = [self.objectify(dataset) for dataset in data]
        return data
    
    def load_data(self, path):
        with open(path) as jsonFile:
            data = json.load(jsonFile)
        return data
    
    def objectify(self, data):
        return [SimpleNamespace(**ex) for ex in data]

    def get_path(self, data_src):
        if data_src == 'wiki_small': paths = self.wiki_small_paths()
        if data_src == 'wiki': paths = self.wiki_paths()
        if data_src == 'wsj':  paths = self.wsj_paths()
        if data_src == 'gcdc':  paths = self.gcdc_paths()
        return paths
    
    def wiki_paths(self):
        base_dir = '/home/alta/Conversational/OET/al826/2021/data/unlabeled'
        paths = [f'{base_dir}/wiki_{i}.json' for i in ['train', 'dev', 'test']]
        return paths
    
    def wiki_small_paths(self):
        paths = self.wiki_paths()
        paths[0] = '/home/alta/Conversational/OET/al826/2021/data/unlabeled/wiki_small.json'
        return paths
    
    def wsj_paths(self):
        base_dir = '/home/alta/Conversational/OET/al826/2021/data/coherence/WSJ'
        paths = [f'{base_dir}/WSJ_{i}.json' for i in ['train', 'dev', 'test']]
        return paths

    def gcdc_paths(self):
        base_dir = "/home/alta/Conversational/OET/al826/2021/data/coherence/GCDC"
        domains = ['clinton', 'enron', 'yelp', 'yahoo']
        sets = ['train', 'test']
        paths = [f'{base_dir}/{domain}_{set1}.json' for domain in domains for set1 in sets]
        return paths