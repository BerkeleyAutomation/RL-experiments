import rlpy.Tools.results as rt

paths = {"Control": "./Results/Experiments/GridWorld/ControlSegment",
         "Segmented Reward": "./Results/Experiments/GridWorld/RewardSegment"}

merger = rt.MultiExperimentResults(paths)
fig = merger.plot_avg_sem("learning_steps", "return")
rt.save_figure(fig, "./Results/Experiments/GridWorld/plot.png")