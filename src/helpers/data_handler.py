from types import SimpleNamespace
import json
import copy
import time
import numpy as np

class DataHandler():
    def __init__(self, data_src):
        if data_src in ['wsj']:
            self.train, self.dev, self.test = self.get_data(data_src)

        if data_src in ['wiki', 'wiki_small']:
            self.train, self.dev, self.test = self.get_data('wiki')
            if data_src == 'wiki_small':
                self.train = self.train[:100_000]
            self.dev = self.dev[:5_000]
            self.test = self.test[:5_000]

        if data_src in ['wiki_debug']:
            self.train, self.dev, self.test = self.get_data('wiki_small')
            self.train = self.train[:1_000]
            self.dev = self.dev[:200]
            self.test = self.test[:200]

        if data_src in ['wiki_unfiltered']:
            self.train = self.get_data('wiki_unfiltered')[0]

        if data_src in ['gcdc']:
            data = self.get_data('gcdc')
            train, test = data[:4], data[4:]
            domains = ['clinton', 'enron', 'yelp', 'yahoo']
            
            for domain, data in zip(domains, train):
                train_name = f'{domain}_train'
                dev_name = f'{domain}_dev'
                setattr(self, train_name, data[:900].copy())
                setattr(self, dev_name, data[900:].copy())
    
            for domain, data in zip(domains, test):
                test_name = f'{domain}_test'
                setattr(self, test_name, data)
            
            self.train = self.clinton_train + self.enron_train + self.yelp_train + self.yahoo_train
            self.dev = self.clinton_dev + self.enron_dev + self.yelp_dev + self.yahoo_dev
            self.test = self.clinton_test + self.enron_test + self.yelp_test + self.yahoo_test
            
    def get_data(self, data_src):
        paths = self.get_paths(data_src)
        data = [self.load_data(path) for path in paths]
        data = [self.objectify(dataset) for dataset in data]
        return data
    
    def load_data(self, path):
        with open(path) as jsonFile:
            data = json.load(jsonFile)
        return data
    
    def objectify(self, data):
        return [SimpleNamespace(**ex) for ex in data]

    def get_paths(self, data_src):
        if data_src == 'wiki_small': paths = self.wiki_small_paths()
        if data_src == 'wiki': paths = self.wiki_paths()
        if data_src == 'wiki_unfiltered': paths = self.wiki_unfiltered_paths()
        if data_src == 'wsj':  paths = self.wsj_paths()
        if data_src == 'gcdc':  paths = self.gcdc_paths()
        return paths
    
    def wiki_paths(self):
        base_dir = '/home/alta/Conversational/OET/al826/2021/data/unlabelled'
        paths = [f'{base_dir}/wiki_{i}.json' for i in ['train', 'dev', 'test']]
        return paths
    
    def wiki_small_paths(self):
        paths = self.wiki_paths()
        paths[0] = '/home/alta/Conversational/OET/al826/2021/data/unlabelled/wiki_small.json'
        return paths
    
    def wiki_unfiltered_paths(self):
        base_dir = '/home/alta/Conversational/OET/al826/2021/data/unlabelled'
        path = [f'{base_dir}/wiki_unfiltered.json']
        return path
    
    def wsj_paths(self):
        base_dir = '/home/alta/Conversational/OET/al826/2021/data/coherence/WSJ'
        paths = [f'{base_dir}/WSJ_{i}.json' for i in ['train', 'dev', 'test']]
        return paths

    def gcdc_paths(self):
        base_dir = "/home/alta/Conversational/OET/al826/2021/data/coherence/GCDC"
        paths = []
        for set_ in ['train', 'test']:
            for domain in ['clinton', 'enron', 'yelp', 'yahoo']:
                paths.append(f'{base_dir}/{domain}_{set_}.json')
        return paths