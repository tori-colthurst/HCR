import numpy as np
from scipy.signal import find_peaks
from scipy import signal
import matplotlib.pyplot as plt
import math

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
			   "LShoulder": 5, "LElbow": 6, "LWrist": 7}

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
			   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"] ]

prev_beat = False
stationary = False
y_dir = 1
x_dir = 0
stationary = False

def check_if_beat(right_hand, curr_frame_idx):
	global prev_beat, x_dir, y_dir, stationary
	beat = False
	threshold = 10
	if curr_frame_idx <= 2:
		return False

	delta_x = right_hand[curr_frame_idx - 1][0] - right_hand[curr_frame_idx - 2][0]
	delta_y = right_hand[curr_frame_idx - 1][1] - right_hand[curr_frame_idx - 2][1]
	delta_x2 = right_hand[curr_frame_idx - 2][0] - right_hand[curr_frame_idx - 3][0]
	delta_y2 = right_hand[curr_frame_idx - 2][1] - right_hand[curr_frame_idx - 3][1]

	if delta_x == 0 and delta_y == 0 and delta_x2 != 0 and delta_y2 != 0:
		# don't update pre_stationary_x/y
		# update stationary global variable
		stationary = False
	elif delta_x == 0 and delta_y == 0 and delta_x2 == 0 and delta_y2 == 0 and prev_beat == False:
		stationary = True
		# beat = True
	# elif delta_x == 0 and delta_y == 0 and delta_x2 == 0 and delta_y2 == 0 and prev_beat == True:
	# 	# prev_beat == True
		# beat = False
	else:
		x_change = False
		y_change = False
		print("Stationary = {}, beat = {}, x_dir = {}, y_dir = {}, delta_x = {}, delta_y = {}".format(stationary, prev_beat, x_dir, y_dir, delta_x, delta_y))
		# Check if prev_stationary_point && delta x/y is in diff direction from dir_xy according to beat map
		# if true, return true and update dir_xy
		# if false, don't do anything
		if stationary and not prev_beat:
			if delta_x <= -threshold and x_dir != -1:
				x_change = True
				x_dir = -1
			elif delta_x >= threshold and x_dir != 1:
				x_change = True
				x_dir = 1

			if delta_y <= -threshold and y_dir != -1:
				y_change = True
				y_dir = -1
			elif delta_x >= threshold and y_dir != 1:
				y_change = True
				y_dir = 1

			if x_change or y_change:
				if delta_x == 0:
					x_dir = 0
				if delta_y == 0:
					y_dir = 0
				prev_beat = True
				beat = True

			# if delta_x <= 1 and x_dir != -1 and delta_y > -1 and y_dir != 1:
			# 	beat = True
			# 	x_dir = -1
			# 	y_dir = 1
			# elif delta_x >= -1 and x_dir != 1 and -1 <= delta_y <= 1 and y_dir != 0:
			# 	beat = True
			# 	x_dir = 1
			# 	y_dir = 0
			# elif delta_x <= 1 and x_dir != -1 and delta_y >= -1 and y_dir != 1:
			# 	beat = True
			# 	x_dir = -1
			# 	y_dir = 1
			# elif -1 <= delta_x <= 1 and x_dir != 0 and delta_y <= 1 and y_dir != -1:
			# 	beat = True
			# 	x_dir = 0
			# 	y_dir =

			# # Beat 1 of 4/4 - bottom to left (beat 2), y increases, x decreases
			# if delta_x <= 1 and x_dir == 0 and delta_y > 0 and y_dir == -1:
			# 	beat = True
			# 	y_dir = 1
			# 	x_dir = -1
			# # Beat 2 of 4/4 - left to right movement
			# elif delta_x >= -1 and x_dir == -1 and -1 <= delta_y <= 1 and y_dir == 1:
			# 	beat = True
			# 	y_dir = 0
			# 	x_dir = 1
			# elif delta_x <= 1 and x_dir == 1 and delta_y >= -1 and y_dir == 0:
			# 	beat = True
			# 	y_dir = 1
			# 	x_dir = -1
			# elif -1 <= delta_x <= 1 and x_dir == -1 and delta_y <= 1 and y_dir == 1:
			# 	beat = True
			# 	y_dir = -1
			# 	x_dir = 0
		else:
			prev_beat = False
			beat = False

		stationary = False

	return beat

def compile_data(points, curr_frame_idx):
	global prev_beat, x_dir, y_dir, stationary
	right_hand = []
	for i in range(0, curr_frame_idx):
		right_hand.append(points[i][BODY_PARTS["RWrist"] - 1])

	print(curr_frame_idx)
	return check_if_beat(right_hand, curr_frame_idx)
