from AlphaGo.models.policy import CNNPolicy
from AlphaGo.ai import GreedyPolicyPlayer
from AlphaGo.ai import ProbabilisticPolicyPlayer
from AlphaGo.go import GameState
from AlphaGo.util import save_gamestate_to_sgf
import AlphaGo.go as go
import uuid


def playout(player1, player2, bd_size=19, print_game=False, player1_name='Player 1', player2_name='Player 2', save_dir='/alphago/playouts', save_name='record.sgf'):
	gamestate = GameState(size=bd_size, enforce_superko=False)
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
			return 0

	print counter
	if print_game:
		save_gamestate_to_sgf(gamestate, save_dir, save_name, player1_name, player2_name)

	return gamestate.get_winner()


model = '/alphago/SLv1/my_model.json'
player1_weights = '/alphago/competition/weights.00006-gold-third-.002lr.hdf5'
#player2_weights = '/alphago/competition/weights.00006-gold-third-.002lr.hdf5'
player2_weights = '/alphago/competition/weights.community.01088.hdf5'
player1_policy = CNNPolicy.load_model(model)
player1_policy.model.load_weights(player1_weights)
player2_policy = CNNPolicy.load_model(model)
player2_policy.model.load_weights(player2_weights)

player1 = GreedyPolicyPlayer(player1_policy)
player2 = GreedyPolicyPlayer(player2_policy)

print playout(player1, player2, print_game=True, save_name="as_black.sgf")
print playout(player2, player1, print_game=True, save_name="as_white.sgf")

playouts = 150

player1 = ProbabilisticPolicyPlayer(player1_policy, temperature=.2)
player2 = ProbabilisticPolicyPlayer(player2_policy, temperature=.2)
player1_wins = 0.0
player2_wins = 0.0
for i in range(0, playouts):
	filename = str(uuid.uuid4()) + '.sgf'
	winner = playout(player1, player2, player1_name='player1', player2_name='player2', print_game=False, save_name=filename)
	print winner
	if winner == go.BLACK:
		player1_wins += 1
	else:
		player2_wins += 1
	filename = str(uuid.uuid4()) + '.sgf'
	winner = playout(player2, player1, player1_name='player2', player2_name='player1', print_game=False, save_name=filename)
	print winner
	if winner == go.WHITE:
		player1_wins += 1
	else:
		player2_wins += 1

print player1_weights
print player1_wins
print player2_weights
print player2_wins
# right now a draw will count as win for opponent. thinking rare enough to not matter
print player1_wins / (playouts * 2.0)
print player2_wins / (playouts * 2.0)



