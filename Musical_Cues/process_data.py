import numpy as np
from scipy.signal import find_peaks
from scipy import signal
import matplotlib.pyplot as plt
import math

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
			   "LShoulder": 5, "LElbow": 6, "LWrist": 7}

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
			   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"] ]

# beat_start = False

def check_if_beat(right_hand, curr_frame_idx):
	global beat_start
	if curr_frame_idx <= 3:
		return False

	delta_x = right_hand[curr_frame_idx - 1][0] - right_hand[curr_frame_idx - 2][0]
	delta_y = right_hand[curr_frame_idx - 1][1] - right_hand[curr_frame_idx - 2][1]

	delta_x2 = right_hand[curr_frame_idx - 2][0] - right_hand[curr_frame_idx - 3][0]
	delta_y2 = right_hand[curr_frame_idx - 2][1] - right_hand[curr_frame_idx - 3][1]

	if delta_x == 0 and delta_y == 0 and delta_x2 == 0 and delta_y2 == 0:
		if beat_start == False:
			beat_start = True
			return True
		else:
			return False
	else:
		beat_start = False
		return False

def compile_data(points, curr_frame_idx):
	global beat_start
	beat_start = False
	right_hand = []
	for i in range(0, curr_frame_idx):
		right_hand.append(points[i][BODY_PARTS["RWrist"] - 1])

	print(curr_frame_idx)
	return check_if_beat(right_hand, curr_frame_idx)
