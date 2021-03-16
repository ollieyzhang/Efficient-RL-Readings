import numpy as np
import core as core
import torch

class ReplayBuffer():
    def __init__(self):
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.pobs_buf = []
        self.terminals = []
        self.length = 0

    def store(self,obs,act,rew,obs_,terminal):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.pobs_buf.append(obs_)
        self.terminals.append(terminal)
        self.length +=1

    def sample(self,batch_size):
        indices = np.random.randint(0, self.length, batch_size)

        obs = [self.obs_buf[indice] for indice in indices]
        act = [self.act_buf[indice] for indice in indices]
        rew = [self.rew_buf[indice] for indice in indices]
        obs_ = [self.pobs_buf[indice] for indice in indices]
        term = [self.terminals[indice] for indice in indices]

        dict = {'observation': np.array(obs),
                'action': np.array(act),
                'reward': np.array(rew),
                'observation_': np.array(obs_),
                'terminal': np.array(term)}

        # dict.items() visit every item of dictionary
        return {k: torch.as_tensor(v, dtype=torch.float32).squeeze(0) for k, v in dict.items()}

class SACReplayBuffer():
    def __init__(self,obs_dim,act_dim,max_size):
        self.obs_buf = np.zeros(core.combined_shape(max_size,obs_dim))
        self.act_buf = np.zeros(core.combined_shape(max_size,act_dim))
        self.rew_buf = np.zeros(max_size)
        self.pobs_buf = np.zeros(core.combined_shape(max_size,obs_dim))
        self.terminals = np.zeros(max_size)
        # self.logp_bs = np.zeros(max_size)
        self.ptr,self.max_size,self.current_size = 0, max_size,0

    def store(self,obs,act,rew,obs_,terminal,*logp_b):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.pobs_buf[self.ptr] = obs_
        self.terminals[self.ptr] = terminal
        # self.logp_bs[self.ptr] = logp_b
        self.ptr = (self.ptr+1) % self.max_size
        self.current_size = min(self.current_size+1,self.max_size)

    def sample(self,batch_size):
        indices = np.random.randint(0,self.current_size,batch_size)
        dict = {'observation': self.obs_buf[indices],
                'action': self.act_buf[indices],
                'reward': self.rew_buf[indices],
                'observation_': self.pobs_buf[indices],
                'terminal': self.terminals[indices]}

        #dict.items() visit every item of dictionary
        return {k: torch.as_tensor(v,dtype=torch.float32).squeeze(0) for k,v in dict.items() }