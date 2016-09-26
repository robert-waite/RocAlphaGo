from AlphaGo.training.reinforcement_policy_trainer import run_training
import os
from cProfile import Profile


datadir = '/alphago/debugTemp/prof'
modelfile = os.path.join(datadir, 'model.json')
weights = os.path.join(datadir, 'weights.00000.hdf5')
outdir = os.path.join(datadir, 'rl_output')
stats_file = os.path.join(datadir, 'rww.prof')

profile = Profile()
arguments = (modelfile, weights, outdir, '--learning-rate', '0.001', '--save-every', '100',
             '--game-batch', '64', '--iterations', '3', '--verbose')

profile.runcall(run_training, arguments)
profile.dump_stats(stats_file)
