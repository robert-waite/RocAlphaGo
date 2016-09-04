from AlphaGo import go
import gtp
from gtp import parse_vertex
import sys
import subprocess
from AlphaGo.util import save_gamestate_to_sgf
import uuid
from subprocess import Popen, PIPE, STDOUT

class ExtendedGtpEngine(gtp.Engine):

	def __init__(self, game_obj, name="NeuralZ", version="0.1"):
		gtp.Engine.__init__(self, game_obj, name, version)

	def cmd_time_left(self, arguments):
		pass

	def cmd_place_free_handicap(self, arguments):
		number_of_stones = int(arguments)
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
		self.cmd_set_free_handicap(vertex_string)
		return vertex_string

	def cmd_set_free_handicap(self, arguments):
		moves = arguments.split()
		vertex_list = []
		for move in moves:
			vertex_list.append(parse_vertex(move))
		self._game.place_handicaps(vertex_list)

	def cmd_final_score(self, arguments):
		try:
			sgf_file_name = self._game.get_sgf()
			p = Popen(['gnugo', '--chinese-rules', '--mode', 'gtp', '-l', sgf_file_name], stdout=PIPE, stdin=PIPE, stderr=PIPE)
			out_bytes = p.communicate(input='final_score\n')[0]
			out_text = out_bytes.decode('utf-8')
			return out_text[2:]
		except subprocess.CalledProcessError as e:
			out_bytes = e.output  # Output generated before error
			print out_bytes.decode('utf-8')
			code = e.returncode  # Return code
			print code

	def cmd_final_status_list(self, arguments):
		try:
			sgf_file_name = self._game.get_sgf()
			p = Popen(['gnugo', '--chinese-rules', '--mode', 'gtp', '-l', sgf_file_name], stdout=PIPE, stdin=PIPE, stderr=PIPE)
			out_bytes = p.communicate(input='final_status_list ' + arguments + '\n')[0]
			out_text = out_bytes.decode('utf-8')
			return out_text[2:]
		except subprocess.CalledProcessError as e:
			out_bytes = e.output  # Output generated before error
			print out_bytes.decode('utf-8')
			code = e.returncode  # Return code
			print code
			return ''
	#
	# def cmd_kgs_genmove_cleanup(self, arguments):
	# 	return self.cmd_genmove(arguments)


class GTPGameConnector(object):
	"""A class implementing the functions of a 'game' object required by the GTP
	Engine by wrapping a GameState and Player instance
	"""

	def __init__(self, player):
		self._state = go.GameState()
		self._player = player

	def clear(self):
		self._state = go.GameState(self._state.size)

	def make_move(self, color, vertex):
		# vertex in GTP language is 1-indexed, whereas GameState's are zero-indexed
		try:
			if vertex == gtp.PASS:
				self._state.do_move(go.PASS_MOVE)
			else:
				(x, y) = vertex
				self._state.do_move((x - 1, y - 1), color)
			return True
		except go.IllegalMove:
			return False

	def set_size(self, n):
		self._state = go.GameState(n)

	def set_komi(self, k):
		self._state.komi = k

	def get_move(self, color):
		self._state.current_player = color
		move = self._player.get_move(self._state)
		if move == go.PASS_MOVE:
			return gtp.PASS
		else:
			(x, y) = move
			return (x + 1, y + 1)

	def get_sgf(self):
		filename = str(uuid.uuid4()) + '.sgf'
		save_gamestate_to_sgf(self._state, '/tmp', filename)
		return '/tmp/' + filename

	def place_handicaps(self, vertexes):
		actions = []
		for vertex in vertexes:
			(x, y) = vertex
			actions.append((x - 1, y - 1))
		self._state.place_handicaps(actions)

def run_gtp(player_obj, inpt_fn=None):
	gtp_game = GTPGameConnector(player_obj)
	gtp_engine = ExtendedGtpEngine(gtp_game)
	if inpt_fn is None:
		inpt_fn = raw_input

	sys.stderr.write("GTP engine ready\n")
	sys.stderr.flush()
	while not gtp_engine.disconnect:
		inpt = inpt_fn()
		# handle either single lines at a time
		# or multiple commands separated by '\n'
		try:
			cmd_list = inpt.split("\n")
		except:
			cmd_list = [inpt]
		for cmd in cmd_list:
			engine_reply = gtp_engine.send(cmd)
			sys.stdout.write(engine_reply)
			sys.stdout.flush()
