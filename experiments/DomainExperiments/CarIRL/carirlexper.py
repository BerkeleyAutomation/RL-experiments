#!/usr/bin/env python
"""
Runs experiment with custom domain
"""
__author__ = "Richard Liaw"
import rlpy
from rlpy.Tools import deltaT, clock, hhmmss, getTimeStr
# from .. import visualize_trajectories as visual
import os
import yaml
import shutil
import inspect
import numpy as np
from rlpy.CustomDomains import RCIRL, Encoding, allMarkovReward


def load_yaml(file_path):
    with open(file_path, 'r') as f:
        ret_val = yaml.load(f)
    return ret_val

def run_experiment_params(param_path='./params.yaml'):
    params = type("Parameters", (), load_yaml(param_path))


    def goalfn(state, goal):
        return ( abs(state[3] - goal[3]) < RCIRL.HEADBOUND and
                # and (abs(state[2] - goal[2]) < RCIRL.SPEEDBOUND) and 
                np.linalg.norm(state[:2] - goal[:2]) < RCIRL.GOAL_RADIUS) # cannot vary

    # Load domain
    def encode_trial():
        rewards = list(params.domain_params['goalArray'])
        encode = Encoding(rewards[1::3], goalfn)
        return encode.strict_encoding

    params.domain_params['goalfn'] = goalfn
    params.domain_params['encodingFunction'] = encode_trial()
    params.domain_params['goalArray'] = params.domain_params['goalArray'][::4]
    domain = eval(params.domain)(**params.domain_params)

    #Load Representation
    representation = eval(params.representation)(
                domain, 
                **params.representation_params)
    policy = eval(params.policy)(
                representation, 
                **params.policy_params)
    agent = eval(params.agent)(
                policy, 
                representation,
                discount_factor=domain.discount_factor, 
                **params.agent_params)

    import ipdb; ipdb.set_trace()

    opt = {}
    opt["exp_id"] = params.exp_id
    opt["path"] = params.results_path + getTimeStr() + "/"
    opt["max_steps"] = params.max_steps
    # opt["max_eps"] = params.max_eps

    opt["num_policy_checks"] = params.num_policy_checks
    opt["checks_per_policy"] = params.checks_per_policy

    opt["domain"] = domain
    opt["agent"] = agent

    if not os.path.exists(opt["path"]):
        os.makedirs(opt["path"])

    shutil.copy(param_path, opt["path"] + "params.yml")
    shutil.copy(inspect.getfile(eval(params.domain)), opt["path"] + "domain.py")
    shutil.copy(inspect.getfile(eval(params.domain)), opt["path"] + "exper.py")


    return eval(params.experiment)(**opt)


if __name__ == '__main__':
    import sys
    experiment = run_experiment_params(sys.argv[1])
    experiment.run(visualize_steps=False,  # should each learning step be shown?
                   visualize_learning=False,
                   visualize_performance=False)  # show policy / value function?
                   # saveTrajectories=False)  # show performance runs?

    experiment.domain.showLearning(experiment.agent.representation)

    # experiment.plotTrials(save=True)
    # experiment.plot(save=True, x = "learning_episode") #, y="reward")
    experiment.save()

