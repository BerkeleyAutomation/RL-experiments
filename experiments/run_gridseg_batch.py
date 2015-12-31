from rlpy.Tools.run import run

run("gridworld_segment.py", "./Results/Experiments/GridWorld/ControlSegment",
    ids=range(5), parallelization="joblib")

run("gridworld_incrsegment.py", "./Results/Experiments/GridWorld/RewardSegment",
    ids=range(5), parallelization="joblib")