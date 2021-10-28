import torch
from torch.nn.functional import logsigmoid, cross_entropy, mse_loss

def select_optimizer(model, optimizer, lr):
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif config.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    return optimizer

def select_loss(loss):    
    def ranking(y1, y2):
        log_likelihood = logsigmoid(y1-y2)
        loss =  -1*torch.mean(log_likelihood)
        return loss
    
    def classification(y1, y2):
        log_likelihood = logsigmoid(y1) + logsigmoid(-1*y2)
        loss =  -1*torch.mean(log_likelihood)/2
        return loss       
    
    def cross_loss(y, targets):
        loss = cross_entropy(y, targets)
        return loss
    
    def MSE(y, targets):
        loss = mse_loss(y, targets)
        return loss
    
    if loss == 'classification':   func = classification
    elif loss == 'ranking':        func = ranking
    elif loss == 'cross':          func = cross_loss
    elif loss == 'mse':            func = MSE
    return func


def triangle_scheduler(SGD_steps):
    def func(i):
        if i <= SGD_steps/10:
            output = 10*i/SGD_steps
        else:
            output = 1 - ((i - 0.1*SGD_steps)/(0.9*SGD_steps))
        return output
    return func