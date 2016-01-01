#!/usr/bin/env python
"""
Runs experiment with custom domain - 9x9-2Path0.txt
"""
__author__ = "Richard Liaw"
from rlpy.CustomDomains import GridWorldInter, GridWorld
from rlpy.Agents import SARSA, Q_Learning
from rlpy.Representations import Tabular
from rlpy.Policies import eGreedy
from rlpy.Experiments import ExperimentSegment, Experiment
from rlpy.Tools import deltaT, clock, hhmmss, getTimeStr
# from .. import visualize_trajectories as visual
import os

maze = os.path.join(GridWorldInter.default_map_dir, '9x9-2PathR2.txt') 

def make_experiment(exp_id=1, path="./Results/Experiments/"):
    """
    Each file specifying an experimental setup should contain a
    make_experiment function which returns an instance of the Experiment
    class with everything set up.

    @param id: number used to seed the random number generators
    @param path: output directory where logs and results are stored
    """
    opt = {}
    opt["exp_id"] = exp_id
    opt["path"] = path

    ## Domain:
    domain = GridWorldInter(maze, noise=0.01)
    # domain.showDomain(s=[0, 2])
    opt["domain"] = domain

    ## Representation
    # discretization only needed for continuous state spaces, discarded otherwise
    representation  = Tabular(domain, discretization=20)

    ## Policy
    policy = eGreedy(representation, epsilon=0.1) ## Need to change this back, limiting noise ATM

    ## Agent
    opt["agent"] = Q_Learning(representation=representation, policy=policy,
                   discount_factor=domain.discount_factor,
                       initial_learn_rate=0.3)
    opt["checks_per_policy"] = 50
    opt["max_steps"] = 5000
    opt["num_policy_checks"] = 50
    experiment = ExperimentSegment(**opt)
    return experiment


# ## DEBUG
# import ipdb; ipdb.set_trace()
# ########


if __name__ == '__main__':
    dirname, filename = os.path.split(os.path.abspath(__file__))

    experiment = make_experiment(1, path=dirname +"/Results/Experiments/9x9_2PathR1/" + getTimeStr() + "/")
    experiment.run(visualize_steps=False,  # should each learning step be shown?
                   visualize_learning=False,  # show policy / value function?
                   saveTrajectories=False)  # show performance runs?

    # visual.saveDomain(experiment, GridWorldInter, maze)
    # import ipdb; ipdb.set_trace()
    experiment.domain.showLearning(experiment.agent.representation)


    # experiment.plotTrials(save=True)
    experiment.saveWeights()
    experiment.plot(save=True) #, y="reward")
    experiment.save()

