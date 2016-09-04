from multiprocessing import Pool, freeze_support
import math
import numpy as np
import os
from AlphaGo.preprocessing.game_converter import game_converter
from AlphaGo.util import save_gamestate_to_sgf
from AlphaGo.util import get_phase
import uuid


BOARD_TRANSFORMATIONS = {
	"noop": lambda feature: feature,
	"rot90": lambda feature: np.rot90(feature, 1),
	"rot180": lambda feature: np.rot90(feature, 2),
	"rot270": lambda feature: np.rot90(feature, 3),
	"fliplr": lambda feature: np.fliplr(feature),
	"flipud": lambda feature: np.flipud(feature),
	"diag1": lambda feature: np.transpose(feature),
	"diag2": lambda feature: np.fliplr(np.rot90(feature, 1))
}

def _walk_all_sgfs(root):
	"""a helper function/generator to get all SGF files in subdirectories of root
	"""
	for (dirpath, dirname, files) in os.walk(root):
		for filename in files:
			if _is_sgf(filename):
				# yield the full (relative) path to the file
				yield os.path.join(dirpath, filename)

def _is_sgf(fname):
		return fname.strip()[-4:] == ".sgf"

def one_hot_action(action, size=19):
	"""Convert an (x,y) action into a size x size array of zeros with a 1 at x,y
	"""
	categorical = np.zeros((size, size))
	categorical[action] = 1
	return categorical

def do_run(sgf_files, phase):
	feature_list = [
		"board",
		"ones",
		"turns_since",
		"liberties",
		"capture_size",
		"self_atari_size",
		"liberties_after",
		"sensibleness",
		"zeros"]
	converter = game_converter(feature_list)
	return converter.sgfs_to_special(sgf_files, phase)

def generate_minibatch_pairs(sgf_list):
	# pool = Pool(processes=16)
	#
	# results = []
	# for i in range(0, 16):
	#     result = pool.apply_async(do_run, (sgf_list[:64], 0))
	#     results.append(result)
	#
	# output = []
	# for result in results:
	#     pair = result.get(timeout=10)
	#     output.extend(pair)

	sub_sets = []
	for i in range(0, len(sgf_list), 64):
		sub_sets.append(sgf_list[i:i + 64])

	output = []
	for i in range(0, 16):
		pair = do_run(sub_sets[i], i)
		output.extend(pair)

	return output

#todo why symmetries change shape of output
#todo seems like broken... is data really correctly formatted?
#todo why generator sometimes returns null
def game_phase_batch_generator(sgf_directory, batch_size, symmetries):
	sgfs = list(_walk_all_sgfs(sgf_directory))
	state_batch_shape = (256, 46, 19, 19)
	#state_batch_shape = (batch_size,) + state_dataset.shape[1:]
	game_size = state_batch_shape[-1]
	Xbatch = np.zeros(state_batch_shape)
	Ybatch = np.zeros((batch_size, game_size * game_size))
	batch_idx = 0
	while True:
		subset_sgfs = np.random.choice(sgfs, 1024)
		pairs = generate_minibatch_pairs(subset_sgfs)
		for pair in pairs:
			# choose a random transformation of the data (rotations/reflections of the board)
			transform = np.random.choice(symmetries)
			# get state from dataset and transform it.
			# loop comprehension is used so that the transformation acts on the 3rd and 4th dimensions
			state = np.array([transform(plane) for plane in pair[0][0]])
			# must be cast to a tuple so that it is interpreted as (x,y) not [(x,:), (y,:)]
			action_xy = tuple(pair[1])
			action = transform(one_hot_action(action_xy, game_size))
			Xbatch[batch_idx] = state
			Ybatch[batch_idx] = action.flatten()
			batch_idx += 1
			if batch_idx == 256:
				batch_idx = 0
				yield (Xbatch, Ybatch)


def shuffled_hdf5_batch_generator(state_dataset, action_dataset, indices, batch_size, transforms=[]):
	"""A generator of batches of training data for use with the fit_generator function
	of Keras. Data is accessed in the order of the given indices for shuffling.
	"""
	state_batch_shape = (batch_size,) + state_dataset.shape[1:]
	game_size = state_batch_shape[-1]
	Xbatch = np.zeros(state_batch_shape)
	Ybatch = np.zeros((batch_size, game_size * game_size))
	batch_idx = 0
	while True:
		for data_idx in indices:
			# choose a random transformation of the data (rotations/reflections of the board)
			transform = np.random.choice(transforms)
			# get state from dataset and transform it.
			# loop comprehension is used so that the transformation acts on the 3rd and 4th dimensions
			state = np.array([transform(plane) for plane in state_dataset[data_idx]])
			# must be cast to a tuple so that it is interpreted as (x,y) not [(x,:), (y,:)]
			action_xy = tuple(action_dataset[data_idx])
			action = transform(one_hot_action(action_xy, game_size))
			Xbatch[batch_idx] = state
			Ybatch[batch_idx] = action.flatten()
			batch_idx += 1
			if batch_idx == batch_size:
				batch_idx = 0
				yield (Xbatch, Ybatch)


def shuffled_hdf5_batch_generator(state_dataset, action_dataset, indices, batch_size, transforms=[]):
	"""A generator of batches of training data for use with the fit_generator function
	of Keras. Data is accessed in the order of the given indices for shuffling.
	"""
	state_batch_shape = (batch_size,) + state_dataset.shape[1:]
	game_size = state_batch_shape[-1]
	Xbatch = np.zeros(state_batch_shape)
	Ybatch = np.zeros((batch_size, game_size * game_size))
	batch_idx = 0
	while True:
		for data_idx in indices:
			# choose a random transformation of the data (rotations/reflections of the board)
			transform = np.random.choice(transforms)
			# get state from dataset and transform it.
			# loop comprehension is used so that the transformation acts on the 3rd and 4th dimensions
			state = np.array([transform(plane) for plane in state_dataset[data_idx]])
			# must be cast to a tuple so that it is interpreted as (x,y) not [(x,:), (y,:)]
			action_xy = tuple(action_dataset[data_idx])
			action = transform(one_hot_action(action_xy, game_size))
			Xbatch[batch_idx] = state
			Ybatch[batch_idx] = action.flatten()
			batch_idx += 1
			if batch_idx == batch_size:
				batch_idx = 0
				yield (Xbatch, Ybatch)


def tmp():
	sgfs = list(_walk_all_sgfs('C:\Users\puff\Desktop\Test Sgfs'))
	subset_sgfs = np.random.choice(sgfs, 1024)
	pairs = generate_minibatch_pairs(subset_sgfs)
	nppairs = np.array(pairs[0][0])
	state_batch_shape = (256,) + nppairs.shape[1:]

	print state_batch_shape

def main():

	# pool = Pool(processes=16)
	#
	# results = []
	# for i in range(0,16):
	#     res = pool.apply_async(get_pairs, ("filename",))
	#     results.append(res)
	#
	# output = []
	# for result in results:
	#     output.append(result.get(timeout=10))
	#
	# print output
	#
	# print get_phase([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 3, 2)

	#generate_minibatch_pairs()
	#tmp()
	# sgf_list = range(0,64)
	#
	# for i in range(0, len(sgf_list), 8):
	# 	print sgf_list[i:i + 8]

	# symstring = 'noop,rot90,rot180,rot270,fliplr,flipud,diag1,diag2'
	# symmetries = [BOARD_TRANSFORMATIONS[name] for name in symstring.strip().split(",")]
	#
	# total = 0
	# for blah in game_phase_batch_generator('C:\Users\puff\Desktop\Test Sgfs', 256, symmetries):
	# 	print len(blah)
	# 	print len(blah[0])
	# 	print len(blah[1])
	# 	total = total + 1
	# 	if total == 2:
	# 		break

	# for i in range(0,16):
	#     phase_moves = get_phase(range(0, 31), 16, i)
	#     print phase_moves

	# for i in range(0, 16):
	#     phase_moves = get_phase(range(0, 32), 16, i)
	#     print phase_moves

	# for i in range(0, 16):
	#     phase_moves = get_phase(range(0, 33), 16, i)
	#     print phase_moves


if __name__ == '__main__':
	main()