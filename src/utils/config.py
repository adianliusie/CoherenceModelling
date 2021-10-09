import torch

def select_optimizer(model, optimizer, lr):
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif config.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    return optimizer

def select_loss(loss):
    log_sigmoid = torch.nn.functional.logsigmoid
    
    def ranking(y1, y2):
        log_likelihood = log_sigmoid(y1-y2)
        loss =  -1*torch.mean(log_likelihood)
        return loss
    
    def classification(y1, y2):
        log_likelihood = log_sigmoid(y1) + log_sigmoid(-1*y2)
        loss =  -1*torch.mean(log_likelihood)/2
        return loss       
        
    if loss == 'classification':   func = classification
    elif loss == 'ranking':        func = ranking
    return func


def triangle_scheduler(SGD_steps):
    def func(i):
        if i <= SGD_steps/10:
            output = 10*i/SGD_steps
        else:
            output = 1 - ((i - 0.1*SGD_steps)/(0.9*SGD_steps))
    return func