from AlphaGo.models.policy import CNNPolicy
from AlphaGo.ai import GreedyPolicyPlayer
from AlphaGo.ai import ProbabilisticPolicyPlayer
from AlphaGo.go import GameState
from AlphaGo.util import plot_network_output
from AlphaGo.util import save_gamestate_to_sgf
from interface.gtp_wrapper import parse_vertex
import AlphaGo.go as go
import uuid

def get_handicap_vertices(number_of_stones):
	if number_of_stones == 2:
		vertex_string = "D4 Q16"
	elif number_of_stones == 3:
		vertex_string = "D4 Q16 D16"
	elif number_of_stones == 4:
		vertex_string = "D4 Q16 D16 Q4"
	elif number_of_stones == 5:
		vertex_string = "D4 Q16 D16 Q4 K10"
	elif number_of_stones == 6:
		vertex_string = "D4 Q16 D16 Q4 D10 Q10"
	elif number_of_stones == 7:
		vertex_string = "D4 Q16 D16 Q4 D10 Q10 K10"
	elif number_of_stones == 8:
		vertex_string = "D4 Q16 D16 Q4 D10 Q10 K4 K16"
	elif number_of_stones == 9:
		vertex_string = "D4 Q16 D16 Q4 D10 Q10 K4 K16 K10"

	moves = vertex_string.split()
	vertex_list = []
	for move in moves:
		(x, y) = parse_vertex(move)
		vertex_list.append((x - 1, y - 1))
	return vertex_list

def playout(player1, player2, bd_size=19, print_game=False, player1_name='Player 1', player2_name='Player 2', save_dir='/alphago/playouts', save_name='record.sgf'):
	gamestate = GameState(size=bd_size)
	#vertex_list = get_handicap_vertices(7)
	#gamestate.place_handicaps(vertex_list)
	counter = 0
	# Play 10 games
	while True:
		counter += 1
		move = player1.get_move(gamestate)
		gamestate.do_move(move)
		if gamestate.is_end_of_game:
			break
		move = player2.get_move(gamestate)
		gamestate.do_move(move)
		if gamestate.is_end_of_game:
			break
		if counter > 1000:
			print 'found excessively long game'
			break

	if print_game:
		save_gamestate_to_sgf(gamestate, save_dir, save_name, player1_name, player2_name)

	return gamestate.get_winner()


model = '/alphago/SLv1/my_model.json'
player1_weights = '/alphago/competition/weights.00006-gold-third-.002lr.hdf5'
player2_weights = '/alphago/competition/weights.00740-wrongu.hdf5'
player1_policy = CNNPolicy.load_model(model)
player1_policy.model.load_weights(player1_weights)
#policy_function = player1_policy.eval_state
player2_policy = CNNPolicy.load_model(model)
player2_policy.model.load_weights(player2_weights)

player1 = GreedyPolicyPlayer(player1_policy)
player2 = GreedyPolicyPlayer(player2_policy)

print playout(player1, player2, print_game=True, save_name="as_black.sgf")
print playout(player2, player1, print_game=True, save_name="as_white.sgf")

playouts = 5

player1 = ProbabilisticPolicyPlayer(player1_policy, temperature=.0001)
player2 = ProbabilisticPolicyPlayer(player2_policy, temperature=.0001)
player1_wins = 0.0
for i in range(0, playouts):
	filename = str(uuid.uuid4()) + '.sgf'
	winner = playout(player1, player2, player1_name='player1', player2_name='player2', print_game=True, save_name=filename)
	print winner
	if winner == go.BLACK:
		player1_wins += 1
	filename = str(uuid.uuid4()) + '.sgf'
	winner = playout(player2, player1, player1_name='player2', player2_name='player1', print_game=True, save_name=filename)
	print winner
	if winner == go.WHITE:
		player1_wins += 1

print player1_wins
print player1_wins / (playouts * 2.0)



