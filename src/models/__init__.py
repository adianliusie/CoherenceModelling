from .bilstm_hier import BilstmHier
from .transformer_flat import TransformerFlat

def select_model(config, T):
    if config.system in ['bert', 'roberta', 'electra']:
        model = TransformerFlat(config.system, config.attention)
        print(f'flat {config.system} transformer being used with attention={config.attention}')
              
    elif config.system in ['glove', 'word2vec']:
        model = BilstmHier(T.embeddings)
        print(f'hierarchical {config.system} BiLSTM system being used')
        
    return model