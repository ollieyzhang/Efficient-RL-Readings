import numpy as np
import copy
import torch
import torch.nn as nn
import itertools
import torch.optim as optim
from buffer import ReplayBuffer,SACReplayBuffer
import core
from spinup.utils.logx import EpochLogger

class SAC(nn.Module):
    def __init__(self,conf,output_dir,exp_name):
        super(SAC, self).__init__()
        self.conf = conf
        self.logger = EpochLogger(output_dir=output_dir, exp_name=exp_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = torch.as_tensor(self.conf.gamma, dtype=torch.float32).to(self.device)
        self.alpha = torch.as_tensor(self.conf.alpha, dtype=torch.float32).to(self.device)

        # how to set random seed so that my result can be reproduced
        self.buffer = SACReplayBuffer(self.conf.obs_dim,self.conf.act_dim,self.conf.buffer_size)
        self.actor = core.Actor(self.conf.obs_dim,self.conf.act_dim,self.conf.act_limit).to(self.device)
        self.critic1 = core.Critic(self.conf.obs_dim,self.conf.act_dim).to(self.device)
        self.critic2 = core.Critic(self.conf.obs_dim,self.conf.act_dim).to(self.device)

        #copy actor and critics
        self.target_critic1 = copy.deepcopy(self.critic1).to(self.device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=self.conf.actor_lr)
        self.critic_paras = itertools.chain(self.critic1.parameters(),self.critic2.parameters())

        self.critic_optimizer = optim.Adam(self.critic_paras,lr=self.conf.critic_lr)

        if self.conf.adjust_temperature:
            if self.conf.target_entropy == 'None':
                self.target_entropy = -np.prod(self.conf.act_dim).item()
            else:
                self.target_entropy = self.conf.target_entropy

            self.target_entropy = torch.as_tensor(self.target_entropy, dtype=torch.float32).to(self.device)
            self.log_alpha = torch.zeros(1, ).to(self.device)
            self.log_alpha.requires_grad = True
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.conf.actor_lr)

        var_counts = tuple(core.count_vars(module) for module in [self.actor, self.critic1, self.critic2])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

        #freeze target network parameters
        for para1, para2 in zip(self.target_critic1.parameters(), self.target_critic2.parameters()):
            para1.requires_grad = False
            para2.requires_grad = False

        # print('\nalpha',self.alpha,' ', self.log_alpha)

    def calculate_target(self,samples):
        obs, act, rew, obs_, done = samples['observation'].to(self.device), samples['action'].to(self.device), samples['reward'].to(self.device), \
                                    samples['observation_'].to(self.device), samples['terminal'].to(self.device)

        with torch.no_grad():
            #sample for policy get next action
            act_, logp_b = self.actor.forward(obs_)
            #get Qvalue using 2 target networks
            q_target_val1 = self.target_critic1.forward(obs_,act_)
            q_target_val2 = self.target_critic2.forward(obs_,act_)
            q_target_val = torch.min(q_target_val1,q_target_val2)

            #calculate target value
            target_val = rew + self.gamma * (1 - done) * (q_target_val - self.alpha * logp_b)

        return target_val

    def critic_loss_func(self,samples):
        obs, act = samples['observation'].to(self.device), samples['action'].to(self.device)

        q_val1 = self.critic1.forward(obs, act)
        q_val2 = self.critic2.forward(obs, act)

        target_val = self.calculate_target(samples)
        critic1_loss = ((q_val1 - target_val)**2).mean()
        critic2_loss = ((q_val2 - target_val)**2).mean()

        #useful info for logging
        #detach() get the value of q_val1, if Q1Vals is changed, q_val1 is also changed!!!
        q_info = dict(Q1Vals=q_val1.cpu().detach().numpy(),
                      Q2Vals=q_val2.cpu().detach().numpy())

        return critic1_loss, critic2_loss, q_info

    def actor_loss_func(self,samples):
        obs = samples['observation'].to(self.device)
        #get action from current policy
        act, logp_b = self.actor.forward(obs)
        if self.conf.adjust_temperature:
            #here detach is used to set required_grad to false
            alpha_loss = -(self.log_alpha * (logp_b + self.target_entropy).detach()).mean()
            self.alpha = self.log_alpha.exp()

            # print('target entropy', self.target_entropy)
            # print('alpha:',self.alpha)

            alpha_info = dict(Alpha=self.alpha.cpu().detach().numpy())
        else:
            alpha_loss = 0
            self.alpha = self.conf.alpha
            alpha_info = None

        q_val1 = self.critic1.forward(obs,act)
        q_val2 = self.critic2.forward(obs,act)
        q_val = torch.min(q_val1,q_val2)

        actor_loss = -(q_val - self.alpha * logp_b).mean()
        # useful info for logging
        # can not use numpy with required gradient varibales
        actor_info = dict(LogPi=logp_b.cpu().detach().numpy())
        return actor_loss, actor_info, alpha_loss, alpha_info

    def update(self,samples):
        # calculate critic1_loss_func and update
        self.critic_optimizer.zero_grad() #!!!
        critic1_loss, critic2_loss, q_info = self.critic_loss_func(samples)
        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()
        self.critic_optimizer.step()

        self.logger.store(LossQ=critic_loss.item(),**q_info)

        # TODO freeze critic parameters when update actor parameters
        for para in self.critic_paras:
            para.requires_grad = False

        # calculate actor_loss_func and update
        self.actor_optimizer.zero_grad()
        self.alpha_optimizer.zero_grad()

        actor_loss, actor_info,alpha_loss, alpha_info = self.actor_loss_func(samples)

        alpha_loss.backward()
        self.alpha_optimizer.step()

        actor_loss.backward()
        self.actor_optimizer.step()

        self.logger.store(LossPi=actor_loss.item(),**actor_info)
        self.logger.store(LossTemp=alpha_loss.item(),**alpha_info)

        #unfreeze critic parameters
        for para in self.critic_paras:
            para.requires_grad = True

        # update target networks
        with torch.no_grad():
            for para1, para2, target_para1, target_para2 in zip(
                    self.critic1.parameters(), self.critic2.parameters(), self.target_critic1.parameters(),
                    self.target_critic2.parameters()):
                target_para1.data.mul_(self.conf.poylak)
                target_para2.data.mul_(self.conf.poylak)
                target_para1.data.add_((1 - self.conf.poylak) * para1.data)
                target_para2.data.add_((1 - self.conf.poylak) * para2.data)

    # get action from current policy
    def get_action(self,obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32).squeeze(0).to(self.device)
        with torch.no_grad():
            act, logp_b = self.actor.forward(obs, deterministic=False)
        return act.cpu().numpy()
