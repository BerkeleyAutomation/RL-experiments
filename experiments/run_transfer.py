import trial_2paths as trials
from numpy import trapz 
import matplotlib.pyplot as plt


"""Prepare sample set of weights to transfer - Setup as follows:
	- Train for 500 steps (or 20 episodes - 1 episode is one full run?), save weights
	- 
Use weights in 3000 steps with evaluations per 100 steps
For Control:
Reward function
AUC: 
For map1:
Reward function
AUC: 
For map2:
Reward function
AUC: 
Should setup pipeline for one continuous method
For map1:
Graph \epsilon vs AUC
Graph environment noise vs AUC
Graph segmented reward vs AUC
Graph step reward vs AUC"""

EPS, ENV_NOISE, SEG_RW, STEP_REW = range(4)
get_AUC = lambda res: trapz(res, dx=1) ##setting dx = 1 but shouldn't matter
 
## save weights after 500 iterations; should be pickled
def prelim_weights(max_steps=500):
	params = {}
	params['max_steps'] = max_steps
	params['num_policy_checks'] = 0 ##may bug out here
	params['domain_class'] = "GridWorld"
	params['mapf'] = "9x9-2Path0.txt"

	trials.run(params)

	##Segmented Trial

	params['domain_class'] = "GridWorldInter"
	params['mapf'] = "9x9-2PathR1.txt"
	trials.run(params)

	## Segmented Trial 2
	params['mapf'] = "9x9-2PathR2.txt"
	trials.run(params)

## run on full experiment on original MDP until convergence
def run_full():
	res1 = ctrl.run(weights=weight_ctrl, num_policy_checks=40, max_steps=4000)
	res2 = ctrl.run(weights=weight_seq1, num_policy_checks=40, max_steps=4000)
	res3 = ctrl.run(weights=weight_seq2, num_policy_checks=40, max_steps=4000)

## may want to save reward data

#FIRST RESULTS?
print "Is res1 > res2? {}".format(res1, res2)

def param_experiment(param, range=None):
	if param == EPS:
		pass
	elif param == ENV_NOISE:
		pass
	elif param == SEG_RW:
		pass
	elif param == STEP_REW:
		pass
	pass

def plot(self, y="return", x="learning_steps", save=False):
    labels = rlpy.Tools.results.default_labels
    performance_fig = plt.figure("Performance")
    res = self.result
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
    if save:
        path = os.path.join(
            self.full_path,
            "{:3}-performance.pdf".format(self.exp_id))
        performance_fig.savefig(path, transparent=True, pad_inches=.1)
    plt.ioff()
    plt.show()

if __name__ == '__main__':
	prelim_weights()
	#run_full()
