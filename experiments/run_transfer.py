from TwoPathExperiments import trial_2pathsreward as trials
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
DEFAULT_PATH = "./TwoPathExperiments/RESETResults/" + getTimeStr() + "/"

def load_results(xpmt_path): ##Should replace this with default library command in rlpy.Tools.results
	results_file = xpmt_path + "_Full/001-results.json"
	with open(results_file, "r") as f:
		results = json.load(f)
	return results

def print_params():
	'''Log the parameters used for this evaluation'''

	pass
 

def run_single_map(singlemap="9x9-2Path0.txt", main_path=DEFAULT_PATH, params=None):
	if params is None:
		params = {}

	params['max_eps'] = 400
	params['num_policy_checks'] = 40

	##DO NOT CHANGE
	params['eval_map'] = "9x9-2Path0.txt"
	######

	params['path'] = main_path + singlemap

	params['mapf'] = singlemap
	# print params
	return trials.run(params)



def param_ranges(param, prange, cur_map=None, params=None):
	if params == None:
		params = {}

	def run_variable(value, params, cur_map):
		# import ipdb; ipdb.set_trace()
		params = params.copy()
		params.update({param: value})
		if cur_map == None:
			return run_single_map(params=params)
		else:
			return run_single_map(cur_map, params=params)

	return [get_AUC(run_variable(value, params, cur_map)["return"]) for value in prange]

def show(variable, vrange, map1="9x9-2PathR1.txt"):
	x = vrange
	ctrl = param_ranges(variable, vrange)
	y = param_ranges(variable, vrange, cur_map=map1)
	try:
		plt.title(variable + " vs AUC")
		plt.ioff()
		plt.plot(x, ctrl)
		plt.plot(x, y)
		plt.legend()
		plt.show()
	finally:
		return ctrl, y

def plot_results(res, y="return", x="learning_episode", exp_name=None, save=False, path="./TwoPathExperiments/TrialResults/"): #TODO
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
	# prelim_weights(max_eps=1000)
	# run_full(max_eps=10000)
	# experiment_plots()
	param_evaluation("STEP_REWARD")
