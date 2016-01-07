import run_transfer as rt
from rlpy.Tools.run import run
from rlpy.Tools.results import default_colors, default_markers, avg_quantity, get_all_result_paths, load_results
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

x_ = [x * 0.05 for x in range (0, 10)] #epsilon/exploration value
x_2 = [x * 0.05 for x in range (0, 10)] ##environment noise
x_3 = [0.02, 0.03, 0.05, 0.06, 0.08, 0.1, 0.2, 0.5, 1, 2, 5] ##door reward
x_4 = [x * -0.001 for x in range(0, 10)] ##step_reward

x_to_y = {"agent_eps": x_,
          "env_noise": x_2,
          "door_reward": x_3,
          "step_reward": x_4
          }

def multiple_parallel_runs(varname, vrange, singlemap="9x9-2Path0.txt"):
    '''Runs 5 seeds of experiment. 
       Results will be saved with special convention (for sake of more than 10 variables)
   '''
    params = rt.default_params()
    params['mapf'] = singlemap
    for i, v in enumerate(vrange):
        params[varname] = v #update hyperparam object
        print params
        print "Running {0}: {1}".format(varname, v)
        run("./TwoPathExperiments/trial_2pathsreward.py", 
            "./TwoPathExperiments/{0}/{1}/{2}".format(varname, singlemap, str('%02d' % i)), 
            ids=range(5), parallelization="joblib", force_rerun=True, **params)


def variable_MER_AUC(var, mp="9x9-2Path0.txt"):
    #LOAD ALL RESULTS 
    results_agent_eps = get_all_result_paths("./TwoPathExperiments/" + var + "/" + mp + "/")
    ##LOAD AS VALUE -> SEED -> RESULT(dict)
    tempres = dict([(x_to_y[var][int(path[-2:])], load_results(path)) for path in results_agent_eps])

    ##Refactor dictionary into -> SEED -> VALUE -> RESULT(dict)
    seed_based_results = defaultdict(dict)
    for val, total_results in tempres.items():
        for seed, result in total_results.items():
            seed_based_results[seed][val] = result

    seed_based_AUC = defaultdict(dict)
    for seed, val_res in seed_based_results.items():
        for val, result in val_res.items():
            seed_based_AUC[seed][val] = rt.get_AUC(result['return'])

    for i, run in seed_based_AUC.items():
        run_values = sorted(run.items(), key=lambda x: x[0])
        keys, values = zip(*run_values)
        seed_based_AUC[i] = {"Values": list(keys), "AUCS": list(values)}

    return seed_based_AUC

def load_variable(var):
    varexp = {"CTRL": variable_MER_AUC(var),
              "MAP1": variable_MER_AUC(var, mp="9x9-2PathR1.txt")}
    return varexp

def plot_avg_sem(
        data, pad_x=False, pad_y=False, xbars=False, ybars=True,
        colors=None, markers=None, xerror_every=1, xscale=None,
        legend=True, **kwargs):
    """
    plots quantity y over x (means and standard error of the mean).
    The quantities are specified by their id strings,
    i.e. "return" or "learning steps"

    :param data: Label->Results

    ``pad_x, pad_y``: if not enough observations are present for some results,
    should they be filled with the value of the last available obervation?\n
    ``xbars, ybars``: show standard error of the mean for the respective 
    quantity colors: dictionary which maps experiment keys to colors.\n
   ``markers``: dictionary which maps experiment keys to markers.
    ``xerror_exery``: show horizontal error bars only every .. observation.\n
    ``legend``: (Boolean) show legend below plot.\n

    Returns the figure handle of the created plot
    """
    x = "Values"
    y = "AUCS"
    style = {
        "linewidth": 2, "alpha": .7, "linestyle": "-", "markersize": 7,
    }
    if colors is None:
        colors = dict([(l, default_colors[i % len(default_colors)])
                      for i, l in enumerate(data.keys())])
    if markers is None:
        markers = dict([(l, default_markers[i % len(default_markers)])
                       for i, l in enumerate(data.keys())])
    style.update(kwargs)
    min_ = np.inf
    max_ = - np.inf
    fig = plt.figure()
    for label, results in data.items():
        style["color"] = colors[label]
        style["marker"] = markers[label]
        y_mean, y_std, y_num = avg_quantity(results, y, pad_y)
        y_sem = y_std / np.sqrt(y_num)
        x_mean, x_std, x_num = avg_quantity(results, x, pad_x)
        x_sem = x_std / np.sqrt(x_num)

        if xbars:
            plt.errorbar(x_mean, y_mean, xerr=x_sem, label=label,
                         ecolor="k", errorevery=xerror_every, **style)
        else:
            plt.plot(x_mean, y_mean, label=label, **style)

        if ybars:
            plt.fill_between(x_mean, y_mean - y_sem, y_mean + y_sem,
                             alpha=.3, color=style["color"])
            max_ = max(np.max(y_mean + y_sem), max_)
            min_ = min(np.min(y_mean - y_sem), min_)
        else:
            max_ = max(y_mean.max(), max_)
            min_ = min(y_mean.min(), min_)

    # adjust visible space
    y_lim = [min_ - .1 * abs(max_ - min_), max_ + .1 * abs(max_ - min_)]
    if min_ != max_:
        plt.ylim(y_lim)

    # axis labels
    xlabel = x
    ylabel = y
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    if xscale:
        plt.xscale(xscale)

    if legend:
        box = plt.gca().get_position()
        plt.gca().set_position([box.x0, box.y0 + box.height * 0.2,
                                box.width, box.height * 0.8])
        legend_handle = plt.legend(loc='upper center',
                                   bbox_to_anchor=(0.5, -0.15),
                                   fancybox=True, shadow=True, ncol=2)
    return fig

