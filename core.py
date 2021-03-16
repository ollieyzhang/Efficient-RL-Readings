import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal
import torch.nn.functional as F

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

LOG_STD_MIN, LOG_STD_MAX = -20,2

class Actor(nn.Module):
    def __init__(self,obs_dim,act_dim,act_limit,hidden_size=(256,256),activation=nn.ReLU):
        super().__init__()
        self.net = mlp([obs_dim]+list(hidden_size),activation,activation)
        self.mu_layer = nn.Linear(hidden_size[-1],act_dim)
        self.log_std_layer = nn.Linear(hidden_size[-1],act_dim)
        # as SAC is used for continuous action space, we should control the boundary of action selection for safety
        self.act_limit = act_limit


    def forward(self,obs,deterministic=False,with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std,LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu,std)

        if deterministic:
            pi_action = mu
        else:
            #rsample is reparameterized sample
            pi_action = pi_distribution.rsample()
        if with_logprob:
            logp_b = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_b -= (2 * (np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=-1)
        else:
            logp_b = None
        # use tanh function to ensure pi_action from net is limited between 0 and 1
        pi_action = torch.tanh(pi_action)
        # pi_action is transformed to actual command
        pi_action = self.act_limit * pi_action

        return pi_action, logp_b


class Critic(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_size=(256,256),activation=nn.ReLU):
        super().__init__()
        self.net = mlp([obs_dim + act_dim] + list(hidden_size) + [1],activation)

    def forward(self,obs,act):
        #torch.float32 is important, must use it
        q = self.net(torch.cat([obs,act],dim=-1))
        return torch.squeeze(q,-1)
