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


def make_experiment(exp_id=1, path="./Results/Experiments/", domain_class="GridWorld", mapf='9x9-2Path0.txt', 
                    max_steps=5000, num_policy_checks=50, agent_eps=0.1, env_noise=0.1, seg_goal=0.8, step_reward=-0.001 weights=None):
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

    maze = os.path.join(GridWorldInter.default_map_dir, mapf) 

    ## Domain:
    if domain_class == "GridWorld":
        domain = GridWorld(maze, noise=env_noise)
    elif domain_class == "GridWorldInter":
        domain = GridWorldInter(maze, noise=env_noise, new_goal=seg_goal)
        
    opt["domain"] = domain

    ## Representation
    # discretization only needed for continuous state spaces, discarded otherwise
    representation = Tabular(domain, discretization=20)
    if weights is not None:
        assert domain_class == "GridWorld" ## ensure that we are transferring to right class
        representation.weight_vec = weights

    ## Policy
    policy = eGreedy(representation, epsilon=agent_eps) ## Need to change this back, limiting noise ATM

    ## Agent
    opt["agent"] = Q_Learning(representation=representation, policy=policy,
                   discount_factor=domain.discount_factor,
                       initial_learn_rate=0.3)
    opt["checks_per_policy"] = 50
    opt["max_steps"] = max_steps
    opt["num_policy_checks"] = num_policy_checks

    experiment = ExperimentSegment(**opt)
    return experiment

def run(opt, saveWeights=False):
    experiment = make_experiment(**opt)
    experiment.run(visualize_steps=False,  # should each learning step be shown?
                   visualize_learning=False,  # show policy / value function?
                   saveTrajectories=False)  # show performance runs?
    if saveWeights:
        experiment.saveWeights()
    experiment.save()
    return experiment.results


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

