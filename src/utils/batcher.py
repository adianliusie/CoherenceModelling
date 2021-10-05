from collections import namedtuple
from tqdm import tqdm
import torch
import random 

from .corruption import create_corrupted_set

flatten = lambda doc: [word for sent in doc for word in sent]
Batch = namedtuple('Batch', ['ids', 'mask'])

class Batcher:
    def __init__(self, bsz=8, schemes=[1], args=None, max_len=512, U=None):
        self.bsz = bsz
        self.schemes = schemes
        self.args = args
        self.max_len = max_len
        
        self.tokenizer = U.tokenizer
        self.CLS = U.CLS
        self.SEP = U.SEP
        
        self.device = torch.device('cpu')
        
    def make_batches(self, documents, c_num, hier=False):
        coherent = documents.copy()
        random.shuffle(coherent)
        coherent = [self.tokenize_doc(doc) for doc in tqdm(coherent)]
        
        if not hier: coherent = [doc for doc in coherent if len(self.flatten_doc(doc)) < self.max_len] 
        else:  coherent = [doc for doc in coherent if max([len(i) for i in doc]) < self.max_len] 
                    
        cor_pairs = self.corupt_pairs(coherent, c_num)
        batches = [cor_pairs[i:i+self.bsz] for i in range(0,len(cor_pairs), self.bsz)]
        batches = [self.prep_batch(batch, hier) for batch in batches]
        return batches        
    
    def corupt_pairs(self, coherent, c_num):
        incoherent = [create_corrupted_set(doc, c_num, self.schemes, self.args) for doc in coherent]
        examples = []
        for pos, neg_set in zip(coherent, incoherent):
            for neg in neg_set:
                examples.append([pos, neg])
        return examples

    def prep_batch(self, pairs, hier=False):
        if hier == False:
            coherent, incoherent = zip(*pairs)
            coherent = [self.flatten_doc(doc) for doc in coherent]
            incoherent = [self.flatten_doc(doc) for doc in incoherent]

            pos_batch = self.batchify(coherent)
            neg_batch = self.batchify(incoherent)
            return pos_batch, neg_batch
        
        else:
            return [[self.batchify(coh), self.batchify(inc)] for coh, inc in pairs]
        
    def tokenize_doc(self, document):
        return [self.tokenizer(sent).input_ids for sent in document]
    
    def flatten_doc(self, document):
        ids = self.CLS + flatten([sent[1:-1] for sent in document]) + self.SEP
        return ids
    
    def batchify(self, batch):
        max_len = max([len(i) for i in batch])
        ids = [doc + [0]*(max_len-len(doc)) for doc in batch]
        mask = [[1]*len(doc) + [0]*(max_len-len(doc)) for doc in batch]
        ids = torch.LongTensor(ids).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        return Batch(ids, mask)

    def to(self, device):
        self.device = device
