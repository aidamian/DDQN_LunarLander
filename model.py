# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, drop=False, 
                 dueling=False, convs=[], dense=[128]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            drop:  use dropout in model
            dueling: use stream splitting for adv/value 
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.conv_blocks = convs
        self.dense_blocks = dense
        self.dueling = dueling
        self.action_size = action_size

        if len(self.conv_blocks) > 0:      
          self.convs = nn.ModuleList()
          prev_conv = state_size
          for nrch in self.conv_blocks[:-1]:
            self.convs.append(nn.Conv2d(prev_conv, nrch, kernel_size=3, stride=1, padding=1))
            self.convs.append(nn.ReLU())
            self.convs.append(nn.MaxPool2d(2,2))
            prev_conv = nrch

          nrch = self.conv_blocks[-1]
          self.convs.append(nn.Conv2d(prev_conv, nrch, kernel_size=3, stride=1, padding=1))

        if len(self.dense_blocks) > 0:
          self.denses = nn.ModuleList()      
          prev_dense = state_size
          for nrunits in self.dense_blocks:
            self.denses.append(nn.Linear(in_features=prev_dense, out_features=nrunits))
            self.denses.append(nn.ReLU())
            if drop:
              self.denses.append(nn.Dropout(p=0.5))
            prev_dense = nrunits
        
        if self.dueling:
          self.pre_adv_layer = nn.Linear(prev_dense, 256) 
          self.pre_adv_act = nn.ReLU()
          self.pre_val_layer = nn.Linear(prev_dense, 256)
          self.pre_val_act = nn.ReLU()
          self.adv_layer = nn.Linear(256, action_size)
          self.val_layer = nn.Linear(256, 1)
        else:
          self.denses.append(nn.Linear(in_features=prev_dense, out_features=action_size))



    def forward(self, x):
        if len(self.conv_blocks) > 0:
          for _layer in self.convs:
            x = _layer(x)
          x = x.view(x.size(0), -1)
        if len(self.dense_blocks) > 0:      
          for _layer in self.denses:
            x = _layer(x)        
        
        if self.dueling:
          x_adv = self.pre_adv_layer(x)
          x_adv = self.pre_adv_act(x_adv)
          x_adv = self.adv_layer(x_adv)
          #x_adv_mean = x_adv.mean(1).unsqueeze(1)
          #x_adv_mean = x_adv_mean.expand(-1, self.action_size)
          x_adv_mean = x_adv.mean(1, keepdim=True)

          x_val = self.pre_val_layer(x)
          x_val = self.pre_val_act(x_val)
          x_val = self.val_layer(x_val)
          #x_val = x_val.expand(-1, self.action_size)
          x_offseted = x_adv - x_adv_mean
          x = x_val + x_offseted
          
        # final output
        return x

