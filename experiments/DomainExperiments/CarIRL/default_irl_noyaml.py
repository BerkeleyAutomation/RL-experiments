from rlpy.CustomDomains import RCIRL, Encoding
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
    opt["num_policy_checks"] = 3
    opt["checks_per_policy"] = 1
    def goalfn(state, goal):
        return (abs(state[3] - goal[3]) < RCIRL.HEADBOUND 
                and np.linalg.norm(state[:2] - goal[:2]) < RCIRL.GOAL_RADIUS) # cannot vary

    encode = Encoding([[0.1, 0.1, 0, 0.1]], goalfn)

    domain = RCIRL([[0.9, .3, 0, 0.7], [1.2, .5, 0, 1]], episodeCap=1000,
                    encodingFunction=encode.strict_encoding)
    opt["domain"] = domain
    representation = Fourier(domain, order=5) # may run into problem with encoding
    policy = eGreedy(representation, epsilon=0.1)

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
    # experiment.save()
