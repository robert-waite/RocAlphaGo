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


MODEL = '/alphago/SLv1/my_model.json'
WEIGHTS = sys.argv[1]
policy = CNNPolicy.load_model(MODEL)
policy.model.load_weights(WEIGHTS)
policy_function = policy.eval_state

player = GreedyPolicyPlayer(policy)
run_gtp(player)

