import pickle
import os, sys
from rlpy.CustomDomains import GridWorld, GridWorldInter

# full_path = os.path.realpath(__file__)
# path, filename = os.path.split(full_path)

# ename = "Segment"
# folder = ""
# MAIN_DIR = os.path.join(path, "Results/Experiments/" + ename + "/Reward/" + folder)


def saveDomain(exp, domain_class, physical_map):
    domain_representation = {}
    domain_representation['domain'] = domain_class
    domain_representation['map'] = physical_map

    domain_fn = os.path.join(exp.full_path, "map.p")

    with open(domain_fn, "w") as f:
        pickle.dump(domain_representation, f)

def visualize(MD):
    trajs = pickle.load(open(os.path.join(MD, "trajectories.p"), "rb"))
    exp_map = pickle.load(open(os.path.join(MD, "map.p"), "rb"))##Doesn't matter which Gridworld used
    d_class = exp_map['domain']
    domain = d_class(exp_map['map'], noise=0.01)
    domain.showTrajectory(trajs)


if __name__ == '__main__':
    visualize(sys.argv[1])