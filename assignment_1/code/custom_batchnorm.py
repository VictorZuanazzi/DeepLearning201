import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of batch normalization in 
PyTorch.
In contrast to more advanced implementations no use of a running mean/variance 
is made.
You should fill in code into indicated sections.
"""

######################################################################################
# Code for Question 3.1
######################################################################################

class CustomBatchNormAutograd(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for 
  MLPs.
  The operations called in self.forward track the history if the input tensors 
  have the flag requires_grad set to True. The backward pass does not need to 
  be implemented, it   is dealt with by the automatic differentiation provided 
  by PyTorch.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormAutograd object. 
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    """
    super(CustomBatchNormAutograd, self).__init__()
    
    #stores number of neuros
    self.n_neurons = n_neurons
    
    #initinalize batch normalization parameters
    self.gamma = nn.Parameter(torch.ones(self.n_neurons))
    self.beta = nn.Parameter(torch.zeros(self.n_neurons))
    self.epsilon = eps

  def forward(self, input):
    """
    Compute the batch normalization
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    
    TODO:
      Check for the correctness of the shape of the input tensor.
      Implement batch normalization forward pass as given in the assignment.
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """
    
    #check if size matches.
    if input.shape[1] != self.n_neurons:
        raise Exception(f"Size DOES matter! Received {input.shape}, expected {self.n_neurons}")
        
    #batch normalization
    
    #mean
    mu = input.mean(dim=0)
    
    #variance
    var = input.var(dim=0, unbiased = False)
    
    #normalization
    center_input = input - mu
    denominator = var + self.epsilon
    denominator = denominator.sqrt()
    
    in_hat = center_input/denominator
    
    #scale and shift
    out = self.gamma*in_hat + self.beta
    
    return out


######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomBatchNormManualFunction(torch.autograd.Function):
  """
  This torch.autograd.Function implements a functional custom version of the 
  batch norm operation for MLPs.
  Using torch.autograd.Function allows you to write a custom backward function.
  The function will be called from the nn.Module CustomBatchNormManualModule
  Inside forward the tensors are (automatically) not recorded for automatic 
  differentiation since the backward pass is done via the backward method.
  The forward pass is not called directly but via the apply() method. This 
  makes sure that the context objects  are dealt with correctly. 
  Example:
    my_bn_fct = CustomBatchNormManualFunction()
    normalized = fct.apply(input, gamma, beta, eps)
  """

  @staticmethod
  def forward(ctx, input, gamma, beta, eps=1e-5):
    """
    Compute the batch normalization
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants 
          and specifying whether tensors need gradients in backward pass
      input: input tensor of shape (n_batch, n_neurons)
      gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
      beta: mean bias tensor, applied per neuron, shpae (n_neurons)
      eps: small float added to the variance for stability
    Returns:
      out: batch-normalized tensor

    TODO:
      Implement the forward pass of batch normalization
      Store constant non-tensor objects via ctx.constant=myconstant
      Store tensors which you need in the backward pass via 
          ctx.save_for_backward(tensor1, tensor2, ...)
      Intermediate results can be decided to be either recomputed in the 
          backward pass or to be stored for the backward pass. Do not store 
          tensors which are unnecessary for the backward pass to save memory!
      For the case that you make use of torch.var be aware that the flag 
          unbiased=False should be set.
    """
    
    #forward pass
    mu = input.mean(dim=0)
    
    #variance
    var = input.var(dim=0, unbiased = False)
    
    #normalization
    center_input = input - mu
    denominator = var + eps
    denominator = denominator.sqrt()
    
    in_hat = center_input/denominator
    
    #scale and shift
    out = gamma * in_hat + beta
        
    #store constants
    ctx.save_for_backward(gamma, mu, center_input, var, denominator, in_hat)
    ctx.epsilon = eps
    
    return out


  @staticmethod
  def backward(ctx, grad_output):
    """
    Compute backward pass of the batch normalization.
    
    Args:
      ctx: context object handling storing and retrival of tensors and 
          constants and specifying whether tensors need gradients in backward pass
    Returns:
      out: tuple containing gradients for all input arguments
    
    TODO:
      Retrieve saved tensors and constants via ctx.saved_tensors and ctx.constant
      Compute gradients for inputs where ctx.needs_input_grad[idx] is True. 
      Set gradients for other inputs to None. This should be decided dynamically.
    """

    #get dimensions
    batch_size, n_neurons = grad_output.shape
    
    #get useful parameters stored in the forward pass
    gamma, mu, center_input, var, denominator, in_hat = ctx.saved_tensors
    
    #to avoid unnecessary matrix inversions 
    den_inv = 1/denominator
    
    #gradient of the input
    if ctx.needs_input_grad[0]:
        
        grad_in_hat = grad_output * gamma
        
        term_1 = batch_size * grad_in_hat
        term_2 = torch.sum(grad_in_hat, dim=0)
        term_3 = in_hat * torch.sum(grad_in_hat * in_hat, dim=0)
        
        grad_input = (1/batch_size) * den_inv * (term_1 - term_2 - term_3)
        
    else:
        grad_input = None
           
        
    #gradient of gamma
    if ctx.needs_input_grad[0] | ctx.needs_input_grad[1]:
        grad_gamma = torch.sum(torch.mul(grad_output, in_hat), dim=0)
    else:
        grad_gamma = None
        
    #gradient of beta
    if ctx.needs_input_grad[2]:
        grad_beta = grad_output.sum(dim=0)
    else:
        grad_beta = None


    # return gradients of the three tensor inputs and None for the constant eps
    return grad_input, grad_gamma, grad_beta, None



######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomBatchNormManualModule(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for 
  MLPs.
  In self.forward the functional version CustomBatchNormManualFunction.forward 
  is called.
  The automatic differentiation of PyTorch calls the backward method of this 
  function in the backward pass.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormManualModule object.
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormManualModule, self).__init__()

    #save parameters
    self.n_neurons = n_neurons
    self.eps = eps
    
    #Initialize gamma and beta
    self.gamma = nn.Parameter(torch.ones(self.n_neurons, dtype=torch.float))
    self.beta = nn.Parameter(torch.zeros(self.n_neurons, dtype=torch.float))

  def forward(self, input):
    """
    Compute the batch normalization via CustomBatchNormManualFunction
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    
    TODO:
      Check for the correctness of the shape of the input tensor.
      Instantiate a CustomBatchNormManualFunction.
      Call it via its .apply() method.
    """
    #check if size matches.
    if input.shape[1] != self.n_neurons:
        raise Exception(f"Size DOES matter! Received {input.shape}, expected {self.n_neurons}")
        
    batch_normalization = CustomBatchNormManualFunction()
    out = batch_normalization.apply(input, self.gamma, self.beta, self.eps)

    return out
