import gym
import os
import torch
import torch.optim as optim
import time
import numpy as np
import itertools
from train_utils import load_conf
from sac_agent import SAC

class RLTrainer():
    def __init__(self,env_name,conf_path,output_dir,exp_name):
        self.env = gym.make(env_name)
        self.full_conf = load_conf(conf_path)
        self.conf = self.full_conf.train_config
        self.conf.obs_dim = self.env.observation_space.shape[0]
        self.conf.act_dim = self.env.action_space.shape[0]
        self.conf.act_limt = self.env.action_space.high[0]

        #in case, we train rl from trained models
        self.init_epoch = 0
        self.exp_name = exp_name
        # set random seed
        np.random.seed(self.conf.seed)
        torch.manual_seed(self.conf.seed)

        self.agent = SAC(self.conf, output_dir, exp_name)

    def test_agent(self):
        pass

    def log_info(self, epoch, start_time):
        # Log info about epoch
        self.agent.logger.log_tabular('Epoch', epoch+self.init_epoch)
        self.agent.logger.log_tabular('EpRet', with_min_and_max=True)
        # logger.log_tabular('TestEpRet', with_min_and_max=True)
        # logger.log_tabular('EpLen', average_only=True)
        # logger.log_tabular('TestEpLen', average_only=True)
        # logger.log_tabular('TotalEnvInteracts', step)
        # logger.log_tabular('Q1Vals', with_min_and_max=True)
        # logger.log_tabular('Q2Vals', with_min_and_max=True)
        # logger.log_tabular('LogPi', with_min_and_max=True)
        self.agent.logger.log_tabular('Alpha', with_min_and_max=True)
        # logger.log_tabular('LossTemp',average_only=True)
        # logger.log_tabular('LossPi', average_only=True)
        # logger.log_tabular('LossQ', average_only=True)
        self.agent.logger.log_tabular('Time', time.time() - start_time)

    def demo2buffer(self,):
        """
        this function loads demonstrated trajectories, runs them in simulation,
        records obs, act, rew, obs_, done in demo_buffer
        """
        demo_conf = self.full_conf.demo_config
        self.demo_conf = demo_conf
        if demo_conf.load_demo_data:
            from buffer import ReplayBuffer
            from train_utils import load_demo_csv
            self.demo_buffer = ReplayBuffer()

            if demo_conf.demo_buf_dir != 'None':
                print('loading existing demo_buffer')
                self.demo_buffer = torch.load(demo_conf.demo_buf_dir)
            else:
                print('loading demonstrated trajectories to replay buffer-----')
                for i in range(demo_conf.load_N):
                    demo_traj = np.array(load_demo_csv(os.path.join(demo_conf.demo_dir,'trajectory{i}.csv'.format(i=i)),demo_conf.traj_len))
                    demo_action = demo_traj[:-2,:] - demo_traj[1:-1,:]
                    obs = self.env.reset()
                    for j in range(len(demo_action)):
                        act = demo_action[j]
                        obs_, rew, done, _ = self.env.step(act)

                        if j == len(demo_action):
                            done = True
                        self.demo_buffer.store(obs, act, rew, obs_, done)
                        obs = obs_

                torch.save(self.demo_buffer,os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', 'sacfd', 'model',
                                    'demo_buffer.pt'))
            print('done-----')
            print('demo_buffer size: ', self.demo_buffer.length)

    def load_models(self,agent_actor,load_dir):
        """
        :param agent_actor: True/False
        :param demo_actor: True/False
        :param demo_buffer: True/False
        :param load_dir: list of string that indicates the load directory
        """
        if agent_actor:
            # load trained model
            assert len(load_dir == 2)
            print("\nloading RL agent_actor, agent_critics, and agent_buffer-----")
            checkpoint = torch.load(load_dir[0])
            self.agent.actor.load_state_dict(checkpoint['actor_para'])
            self.agent.critic1.load_state_dict(checkpoint['critic1_para'])
            self.agent.critic2.load_state_dict(checkpoint['critic2_para'])
            self.agent.target_critic1.load_state_dict(checkpoint['target_critic1'])
            self.agent.target_critic2.load_state_dict(checkpoint['target_critic2'])

            import copy
            self.agent.log_alpha = copy.deepcopy(checkpoint['log_alpha'])
            self.agent.alpha = self.agent.log_alpha.exp()
            #if we do not have this line, log_alpha wound not be updated
            self.agent.alpha_optimizer = optim.Adam([self.agent.log_alpha], lr=self.agent.actor_lr)
            self.start_epoch = checkpoint['epoch']

            self.agent.buffer = torch.load(load_dir[1])
            print("\ndone-----")


    def train(self,model_name,buffer_name):
        """
        :param model_name: name for saving model. for instance, model_with_demo
        :param buffer_name: name for saving buffer.
        """
        # prepare for interaction with environment
        total_steps = self.conf.steps_per_epoch * self.conf.epochs
        start_time = time.time()
        obs, ep_ret, ep_len = self.env.reset(), 0, 0

        for step in range(total_steps):
            if step > self.conf.start_steps:
                act = self.agent.get_action(obs)
            else:
                # initial phase, action is selected randomly
                act = self.env.action_space.sample()

            obs_, rew, done, _ = self.env.step(act)
            ep_ret += rew
            # print('ep_ret:',ep_ret)
            ep_len += 1
            # print('ep_len:',ep_len)
            done = False if ep_len == self.conf.max_ep_len else done
            self.agent.buffer.store(obs, act, rew, obs_, done)

            obs = obs_

            # max_ep_len refers to the maximum length of each epoch. it also means the max length for one task
            if done or (ep_len == self.conf.max_ep_len):
                self.agent.logger.store(EpRet=ep_ret, EpLen=ep_len)
                obs, ep_ret, ep_len = self.env.reset(), 0, 0

            if (step > self.conf.update_step) and (step % self.conf.update_every == 0):
                for j in range(self.conf.update_every):
                    #if we are running SACfD, then sample from agent.buffer and demo_buffer with the sim_demo_rat
                    if self.demo_conf.load_demo_data:
                        sim_samples = self.agent.buffer.sample(int(self.conf.batch_size * self.demo_conf.sim_demo_rat))
                        demo_samples = self.demo_buffer.sample(int(self.conf.batch_size * self.demo_conf.sim_demo_rat))
                        keys = sim_samples.keys()
                        values = tuple(torch.cat((v1,v2),0) for v1,v2 in zip(sim_samples.values(),demo_samples.values()))
                        samples = dict(zip(keys,values))
                    else:
                        samples = self.agent.buffer.sample(self.conf.batch_size)
                    self.agent.update(samples)

            # each epoch, we should save model and info
            if (step + 1) % self.conf.steps_per_epoch == 0:
                # // is floordiv
                epoch = (step + 1) // self.conf.steps_per_epoch

                # save model
                if (epoch % self.conf.save_freq == 0) or (epoch == self.conf.epochs):
                    self.agent.logger.save_state({'env': self.env}, None)
                    state = {'actor_para': self.agent.actor.state_dict(), 'critic1_para': self.agent.critic1.state_dict(),
                             'critic2_para': self.agent.critic2.state_dict(), 'log_alpha': self.agent.log_alpha, 'epoch': epoch,
                             'target_critic1': self.agent.target_critic1.state_dict(),
                             'target_critic2': self.agent.target_critic2.state_dict()}

                    torch.save(state, os.path.join(os.path.dirname(os.path.abspath(__file__)),'result', self.exp_name,'model', model_name+'.pt'))
                if epoch % 50 == 0:
                    torch.save(self.agent.buffer,os.path.join(os.path.dirname(os.path.abspath(__file__)),'result', self.exp_name,'model', buffer_name+'.pt'))

                self.log_info(epoch,start_time)
                self.agent.logger.dump_tabular()


if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plot')
    conf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'config','version0.yaml')
    import argparse
    import pybullet_envs
    import gym_panda

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',type=str,help='sac or sacfd')
    parser.add_argument('--env_name', type=str, default='panda-v1')
    parser.add_argument('--model_name',type=str,help='enter a distinguish name for saving trained models')
    parser.add_argument('--buffer_name',type=str,help='enter a distinguish name for saving replay buffer ')
    args = parser.parse_args()

    print('\033[1;32;40mnum of threads: %d \033[0m' % torch.get_num_threads())
    torch.set_num_threads(torch.get_num_threads())

    sac_trainer = RLTrainer(args.env_name,conf_dir,output_dir,args.exp_name)

    sac_trainer.demo2buffer()

    sac_trainer.train(args.model_name,args.buffer_name)