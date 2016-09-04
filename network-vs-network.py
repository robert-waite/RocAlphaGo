import numpy as np
import sys
from AlphaGo.models.policy import CNNPolicy
from interface.gtp_wrapper import run_gtp
from AlphaGo.ai import GreedyPolicyPlayer
from AlphaGo.ai import MCTSPlayer
from AlphaGo.go import GameState
from AlphaGo.util import save_gamestate_to_sgf

def policy_network(state):
	moves = state.get_legal_moves(include_eyes=False)
	# 'random' distribution over positions that is smallest
	# at (0,0) and largest at (18,18)
	probs = np.arange(361, dtype=np.float)
	probs = probs / probs.sum()
	return zip(moves, probs)

def value_network(state):
	# it's not very confident
	return 0.0

def rollout_policy(state):
	# just another policy network
	return policy_network(state)

def create_greedy_player(model_file, weights_file, ):
	policy = CNNPolicy.load_model(model_file)
	policy.model.load_weights(weights_file)
	return GreedyPolicyPlayer(policy)


MODEL = '/alphago/SLv1/my_model.json'
WEIGHTS = sys.argv[1]

player1 = create_greedy_player(MODEL, sys.argv[1])
player2 = create_greedy_player(MODEL, sys.argv[2])



