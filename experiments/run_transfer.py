from TwoPathExperiments import trial_2paths as trials
from numpy import trapz 
from rlpy.Tools import plt
from collections import defaultdict
import numpy as np
import rlpy.Tools.results 
from rlpy.Tools import getTimeStr
import pickle
import json
import os


'''TODOS:
 - Replace loads with default commands
 - Remove pickle, replace with JSON
 - Save Parameters'''

EPS, ENV_NOISE, SEG_RW, STEP_REW = range(4)
get_AUC = lambda res: trapz(res, dx=1) ##setting dx = 1 but shouldn't matter
DEFAULT_PATH = "./TwoPathExperiments/TrialResults/" + getTimeStr() + "/"
def load_weight(xpmt_path):
	weight_file = xpmt_path + "/weights.p"
	with open(weight_file, "r") as f:
		weights = pickle.load(f)
	return weights

def load_results(xpmt_path): ##Should replace this with default library command in rlpy.Tools.results
	results_file = xpmt_path + "_Full/001-results.json"
	with open(results_file, "r") as f:
		results = json.load(f)
	return results

def print_params():
	'''Log the parameters used for this evaluation'''

	pass
 
## save weights after 500 iterations; should be pickled
def prelim_weights(max_steps=500, main_path=DEFAULT_PATH, params=None):

	if params is None:
		params = {}
	params['max_steps'] = max_steps
	params['num_policy_checks'] = 0 ##may bug out here
	params['domain_class'] = "GridWorld"
	params['mapf'] = "9x9-2Path0.txt"
	params['path'] = main_path + "CTRL"

	trials.run(params, saveWeights=True)

	##Segmented Trial
	params['domain_class'] = "GridWorldInter"
	params['mapf'] = "9x9-2PathR1.txt"
	params['path'] = main_path + "MAP1"
	trials.run(params, saveWeights=True)

	## Segmented Trial 2
	params['mapf'] = "9x9-2PathR2.txt"
	params['path'] = main_path + "MAP2"
	trials.run(params, saveWeights=True)

## run on full experiment on original MDP until convergence
def run_full(max_steps=5000, main_path=DEFAULT_PATH, params=None, extra_trial=True):
	if params is None:
		params = {}

	params['max_steps'] = max_steps
	params['num_policy_checks'] = max_steps / 100
	params['domain_class'] = "GridWorld"
	params['mapf'] = "9x9-2Path0.txt"

	params['path'] = main_path + "CTRL_Full"
	params['weights'] = load_weight(main_path + "CTRL")
	res1 = trials.run(params)

	params['path'] = main_path + "MAP1_Full"
	params['weights'] = load_weight(main_path + "MAP1")
	res2 = trials.run(params)

	res3 = None
	if extra_trial:
		params['path'] = main_path + "MAP2_Full"
		params['weights'] = load_weight(main_path + "MAP2")
		res3 = trials.run(params)

	return [res1, res2, res3]

def param_evaluation(variable):
	var_results = defaultdict()
	def variable_run(var, eval_range):
		path_start = DEFAULT_PATH + var + "/"
		var_results['variable_chosen'] = var
		for x in eval_range:
			path = path_start + "val_{}/".format(x)
			params = {var: x}
			print "Evaluating weights for {0}: {1}".format(var, x)
			prelim_weights(main_path=path, params=params)
			print "==================="
			print "Finished evaluating weights... \n Running full experiment"
			run_full(main_path=path, params=params)
			var_results[x] = experiment_plots(path)

		with open(path_start + "eval_results.json", "w") as f:
			json.dump(var_results, f)

	if variable == "EPS":
		variable_run("agent_eps", [x * 0.05 for x in range (0, 10)])
	elif variable == "ENV_NOISE":
		variable_run("env_noise", [x * 0.05 for x in range (0, 16)])
	elif variable == "SEG_REWARD":
		variable_run("seg_goal", [1 - x*0.05 for x in range(0, 15)]) #sets final reward at goal
	elif variable == "STEP_REWARD":
		variable_run("step_reward", [x * -0.001 for x in range(0, 10)])


def experiment_plots(main_path="./TwoPathExperiments/TrialResults/", show=False):
	def load_print(exp):
		res = load_results(main_path + exp)
		auc = get_AUC(res['return'])
		print "AUC FOR {0}: {1}".format(exp, auc)
		if show:
			plot_results(res, y="return", exp_name=exp, save=True, path=main_path)
		return auc

	return [load_print(x) for x in ["CTRL", "MAP1", "MAP2"]]
	

def plot_results(res, y="return", x="learning_steps", exp_name=None, save=False, path="./TwoPathExperiments/TrialResults/"): #TODO
    labels = rlpy.Tools.results.default_labels
    performance_fig = plt.figure("Performance")
    plt.plot(res[x], res[y], '-bo', lw=3, markersize=10)
    plt.xlim(0, res[x][-1] * 1.01)
    y_arr = np.array(res[y])
    m = y_arr.min()
    M = y_arr.max()
    delta = M - m
    if delta > 0:
        plt.ylim(m - .1 * delta - .1, M + .1 * delta + .1)
    xlabel = labels[x] if x in labels else x
    ylabel = labels[y] if y in labels else y
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    if exp_name is not None:
        plt.title(exp_name)
    if save:
        path = os.path.join(
            path,
            "{:3}-performance.png".format(exp_name))
        performance_fig.savefig(path, transparent=True, pad_inches=.1)
    plt.ioff()
    plt.show()

if __name__ == '__main__':
	# prelim_weights(max_steps=1000)
	# run_full(max_steps=10000)
	# experiment_plots()
	param_evaluation("STEP_REWARD")
