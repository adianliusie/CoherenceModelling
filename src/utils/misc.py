from scipy.stats import norm
import numpy as np
import torch

def gaussian_pdf(mean, variance):
    Z = 1/np.sqrt(2*np.pi*variance)

    def _pdf(x):
        prob = Z*torch.exp(-1*(x-mean)**2/2*variance)
        return prob
    return _pdf
                                                       
"""                                                       
def gaussian_pdf(mean, variance):
    def _gaussian_pdf(x):
        sigma = np.sqrt(variance)
        z = (x-mean)/sigma
        prob = (1/torch.sqrt(2*np.pi*variance))*np.exp(norm.pdf(z)/sigma
        return prob
    
    class GaussianPDF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            y = _gaussian_pdf(x)
            return torch.FloatTensor(y)

        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad = (-2*(x-mean)/variance)*_gaussian_pdf(x)
            return grad_input
     
    return GaussianPDF.apply
"""
                                                       
def toggle_grad(func):
    def inner(*args, no_grad=False):
        if no_grad==True:
            with torch.no_grad():
                return func(*args)
        else:
            return func(*args)
    return inner
