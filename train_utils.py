import numpy as np
import prodict, yaml


def load_conf(path):
    with open(path,'r') as stream:
        yaml_dict = yaml.load(stream,Loader=yaml.FullLoader)
    return prodict.Prodict.from_dict(yaml_dict)

def load_demo_csv(path,length):
    traj = np.loadtxt(path,dtype=np.float,delimiter=',')
    traj = np.array(traj)[:length,:]
    return traj