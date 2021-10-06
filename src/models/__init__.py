from .bilstm_hier import BilstmHier
from .transformer_flat import TransformerFlat
from ..utils.tokenizer import get_embeddings

def select_model(config):
    if config.system in ['bert', 'roberta', 'electra']:
        model = TransformerFlat(config.system, config.attention)
        print(f'\nFLAT {config.system.upper()} TRANSFORMER- attention={config.attention}')
              
    elif config.system in ['glove', 'word2vec']:
        embeddings = get_embeddings(config.system)
        model = BilstmHier(embeddings)
        print(f'hierarchical {config.system} BiLSTM system being used')
        
    return model