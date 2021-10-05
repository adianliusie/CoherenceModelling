from transformers import BertTokenizerFast, RobertaTokenizerFast
from tqdm import tqdm
import torch

class TokenizerClass:
    def __init__(self, system, lim=300_000):
        if system in ['bert', 'electra']:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.CLS, self.SEP = [101], [102]
            self.embeddings = None
            
        elif system == 'roberta':
            self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
            self.CLS, self.SEP = [0], [2]
            self.embeddings = None

        elif system in ['glove', 'word2vec']:
            path = self.get_embedding_path(system)
            tok_dict, embed_matrix = self.read_embeddings(path, lim)
            self.tokenizer = FakeTokenizer(tok_dict)
            self.embeddings = torch.Tensor(embed_matrix)
            self.CLS, self.SEP = [], []

        else:
            raise ValueError('invalid system')
            
    def get_embedding_path(self, name):
        base_dir = '/home/alta/Conversational/OET/al826/2021'
        if name == 'glove': path = f'{base_dir}/data/embeddings/glove.840B.300d.txt'
        elif name == 'word2vec': path = f'{base_dir}/data/embeddings/word2vec.txt'
        else: raise ValueError('invalid word embedding system') 
        return path 

    def read_embeddings(self, path, limit=300_000):
        with open(path, 'r') as file:
            _ = next(file)
            tok_dict = {}
            embed_matrix = []
            for index, line in tqdm(zip(range(limit), file), total=limit):
                word, *embedding = line.split()
                if len(embedding) == 300 and word not in tok_dict:
                    embed_matrix.append([float(i) for i in embedding])
                    tok_dict[word] = len(tok_dict)
        return tok_dict, embed_matrix

#Making the tokenizer the same format as huggingface to better interface with code
class FakeTokenizer:
    def __init__(self, tok_dict):
        self.tok_dict = tok_dict
        self.reverse_dict = {v:k for k,v in self.tok_dict.items()}

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
