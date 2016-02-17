from rlpy.CustomDomains import RCIRL, Encoding, allMarkovReward
from rlpy.Agents import Q_Learning
from rlpy.Representations import *
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {'discretization': hp.quniform("discretization", 5, 50, 1),
               'lambda_': hp.uniform("lambda_", 0., 1.),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(5e-2), np.log(1))}

rewards = [[0.0, 0.0, -0.03, 0.0],
 [-0.01, 0.0, -0.06, 0.02],
 [-0.03, -0.0, -0.09, 0.05],
 [-0.05, -0.0, -0.12, 0.1],
 [-0.09, -0.01, -0.15, 0.17],
 [-0.13, -0.01, -0.18, 0.26],
 [-0.19, -0.03, -0.21, 0.16],
 [-0.25, -0.04, -0.24, 0.03],
 [-0.32, -0.04, -0.27, -0.1],
 [-0.4, -0.03, -0.3, -0.26],
 [-0.49, -0.01, -0.3, -0.43],
 [-0.57, 0.03, -0.3, -0.61],
 [-0.64, 0.08, -0.3, -0.78],
 [-0.71, 0.14, -0.27, -0.95],
 [-0.75, 0.21, -0.24, -1.11],
 [-0.79, 0.27, -0.21, -1.25],
 [-0.81, 0.33, -0.18, -1.37],
 [-0.82, 0.39, -0.15, -1.47],
 [-0.82, 0.43, -0.12, -1.56]]

def make_experiment(
        exp_id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        boyan_N0=238,                                   
        lambda_=0.9,
        initial_learn_rate=.1,
        discretization=20):
    opt = {}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    opt["exp_id"] = exp_id
    opt["path"] = path
    opt["max_steps"] = 15000
    opt["num_policy_checks"] = 5
    opt["checks_per_policy"] = 5
    def goalfn(state, goal):
        return ( abs(state[3] - goal[3]) < RCIRL.HEADBOUND and
                # and (abs(state[2] - goal[2]) < RCIRL.SPEEDBOUND) and 
                np.linalg.norm(state[:2] - goal[:2]) < RCIRL.GOAL_RADIUS) # cannot vary

    def encode_trial():
        encode = Encoding(rewards[1::4], goalfn)
        return encode.strict_encoding

    domain = RCIRL(rewards[::2], episodeCap=1000,
                    encodingFunction=encode_trial(),
                    goalfn=goalfn,
                    step_reward=-0.5)
                    # rewardFunction=allMarkovReward)
    opt["domain"] = domain
    representation = Fourier(domain, order=3) # may run into problem with encoding
    policy = eGreedy(representation, epsilon=0.2)

    opt["agent"] = Q_Learning(
        policy, representation,discount_factor=domain.discount_factor,
        lambda_=0.9, initial_learn_rate=initial_learn_rate,
        learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**opt)
    return experiment

if __name__ == '__main__':
    # run_profiled(make_experiment)
    experiment = make_experiment()
    experiment.run(visualize_steps=True)
    experiment.plot()
    experiment.save()
