import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import numpy as np

def load_reward(filename):
    dataset = []

    with open(filename,'r') as reward_list:
        lines = reward_list.readline()
        while True:
            lines = reward_list.readline()
            if not lines:
                break
            r_list = [float(item) for item in lines.split()]
            dataset.append(r_list[:4])

    dataset = np.array(dataset)
    return dataset

def plot_reward_with_variance(dataset,name,color):
    f,ax = plt.subplots(1,1)
    ax.plot(dataset[:,0],dataset[:,1],color=color)
    r1 = list(map(lambda x: x[0]-x[1],zip(dataset[:,1],dataset[:,2])))
    r2 = list(map(lambda x: x[0]+x[1],zip(dataset[:,1],dataset[:,2])))
    ax.fill_between(dataset[:,0],r1,r2,color=color,alpha=0.2)
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward')
    plt.show()
    f.savefig(os.path.join('Plot',name+'.png'),dpi=100)

def plot_multiple_reward(datasets,legend,name):
    f, ax = plt.subplots(1, 1)
    for _ in range(len(datasets)):
        ax.plot(datasets[_,:, 0], datasets[_,:, 1], color=colors[_])
        r1 = list(map(lambda x: x[0] - x[1], zip(datasets[_,:, 1], datasets[_,:, 2])))
        r2 = list(map(lambda x: x[0] + x[1], zip(datasets[_,:, 1], datasets[_,:, 2])))
        ax.fill_between(datasets[_,:, 0], r1, r2, color=colors[_], alpha=0.2)
    ax.legend(legend)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward')
    plt.show()
    f.savefig(os.path.join('Plot', name + '.png'), dpi=100)

if __name__ == "__main__":
    sns.set_style('whitegrid')
    exp_dir = 'Plot/'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
    else:
        os.makedirs(exp_dir, exist_ok=True)

    #load txt files
    # filename = os.path.join('Plot', 'progress_gpu_with_demo1.txt')
    # dataset = load_reward(filename)
    # color = cm.viridis(0.7)
    # plot_reward_with_variance(dataset,'sac_gpu_with_demo_reward1',color)

    #load two txt files
    filename1 = os.path.join('Plot', 'progress_gpu_s0.txt')
    filename2 = os.path.join('Plot', 'progress.txt')
    dataset1 = load_reward(filename1)
    dataset2 = load_reward(filename2)
    length = min(len(dataset1),len(dataset2))
    datasets = np.array([dataset1[:length,:],dataset2[:length,:]])

    # define colors for each reward list
    colors = [cm.viridis(0.7), cm.magma(0.7)]

    #TODO change the name for saving picture
    plot_multiple_reward(datasets,['SAC','SACfD'],'SACfD_SAC_Comp1')