from AlphaGo.models.policy import CNNPolicy
from AlphaGo.util import sgf_iter_states
from AlphaGo.util import sgf_to_gamestate
from AlphaGo.go import WHITE

model = '/alphago/SLv1/my_model.json'
weights = '/alphago/competition/weights.00740-wrongu.hdf5'
game = '/alphago/debugTemp/superko.sgf'
policy = CNNPolicy.load_model(model)
policy.model.load_weights(weights)

with open(game) as f:
	sgf_string = f.read()

gamestate = sgf_to_gamestate(sgf_string)
gamestate.do_move((1, 18), WHITE)
