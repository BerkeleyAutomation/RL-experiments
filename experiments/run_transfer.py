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
 
def default_params(params=None):
	if params is None:	
		params = {}

	if 'max_eps' not in params:
		params['max_eps'] = 400 

	if 'num_policy_checks' not in params:
		params['num_policy_checks'] = 40

	##DO NOT CHANGE
	params['eval_map'] = "9x9-2Path0.txt"
	######
	return params

def run_single_map(singlemap="9x9-2Path0.txt", main_path=DEFAULT_PATH, params=None):
	params = default_params(params)
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

def show_one_variable(variable, vrange, map1="9x9-2PathR1.txt"):
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

