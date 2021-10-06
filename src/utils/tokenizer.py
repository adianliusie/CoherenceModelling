from transformers import BertTokenizerFast, RobertaTokenizerFast
from tqdm import tqdm
from functools import lru_cache
import torch

def get_tokenizer(system):
    if system in ['bert', 'electra']:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        
    elif system == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')   
        
    elif system in ['glove', 'word2vec']:
        tok_dict, embed_matrix = read_embeddings(system)
        tokenizer = FakeTokenizer(tok_dict)
    return tokenizer

def get_embeddings(system):
    tok_dict, embed_matrix = read_embeddings(system)
    return embed_matrix

@lru_cache(maxsize=2)
def read_embeddings(system):
    path = get_embedding_path(system)
    with open(path, 'r') as file:
        _ = next(file)
        tok_dict = {}
        embed_matrix = []
        for line, _ in tqdm(zip(file, range(500_000)), total=500_000):
            word, *embedding = line.split()
            if len(embedding) == 300 and word not in tok_dict:
                embed_matrix.append([float(i) for i in embedding])
                tok_dict[word] = len(tok_dict)
    return tok_dict, torch.Tensor(embed_matrix)

def get_embedding_path(name):
    base_dir = '/home/alta/Conversational/OET/al826/2021'
    if name == 'glove': path = f'{base_dir}/data/embeddings/glove.840B.300d.txt'
    elif name == 'word2vec': path = f'{base_dir}/data/embeddings/word2vec.txt'
    return path

#Making the tokenizer the same format as huggingface to better interface with code
class FakeTokenizer:
    def __init__(self, tok_dict):
        self.tok_dict = tok_dict
        self.reverse_dict = {v:k for k,v in self.tok_dict.items()}
        self.cls_token_id  = None
        self.sep_token_id = None

    def tokenize_word(self, w):
        if w in self.tok_dict:  output = self.tok_dict[w]
        else: output = len(self.tok_dict)-1
        return output

    def tokenize(self, x):
        tokenized_words = [self.tokenize_word(i) for i in x.split()]
        x = type('TokenizedInput', (), {})()
        setattr(x, 'input_ids', tokenized_words)
        return x

    def decode(self, x):
        return ' '.join([self.reverse_dict[i] for i in x])

    def __call__(self, x):
        return self.tokenize(x)
