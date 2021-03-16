import torch
import core
import csv
import gym
import gym_panda
import os

obs_dim = 3
act_dim = 3
act_limit = 1

#load trained policy
actor_critic_kwargs = dict()
actor = core.Actor(obs_dim,act_dim,act_limit=act_limit,**actor_critic_kwargs)
checkpoint = torch.load(os.path.join('progress','model','model_para_trained1.pt'))
actor.load_state_dict(checkpoint['actor_para'])

env = gym.make('panda-v1')

#get action from current policy
def get_action(obs,deterministic=False):
    with torch.no_grad():
        act, logp_b = actor.forward(obs,deterministic=False)
    return act.numpy()

#record trajectories
trajectory = []
indices = 0
total_steps = 100000
obs, ep_ret, ep_len = env.reset(), 0, 0
trajectory.append(obs)
max_ep_len = 1000

for step in range(total_steps):
    act = get_action(obs)

    obs_, rew, done, _ = env.step(act)
    ep_ret += rew
    ep_len += 1
    done = False if ep_len == max_ep_len else done

    obs = obs_
    trajectory.append(obs)
    # max_ep_len refers to the maximum length of each epoch. it also means the max length for one task
    # print('\nstep: ', step % max_ep_len,'distance:',_)
    if done or (ep_len == max_ep_len):
        print('  ep_ret:',ep_ret,'  ep_len:',ep_len)

        demo_dir = os.path.join('data','demo_trained_rl','trajectory{i}.csv'.format(i=str(indices)))
        ret_dir = os.path.join('data','demo_trained_rl','ep_ret.txt')
        with open(demo_dir,'w',newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in trajectory:
                writer.writerow(row)

        with open(ret_dir,'a',newline='') as file:
            file.write(str(ep_ret))
            file.write('\n')
            
        obs, ep_ret, ep_len = env.reset(), 0, 0
        indices += 1
        trajectory = []

