import numpy as np
from scipy.signal import find_peaks
from scipy import signal
import matplotlib.pyplot as plt
import math

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
			   "LShoulder": 5, "LElbow": 6, "LWrist": 7}

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
			   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"] ]

# right_hand = {}
right_hand = []

def graph_data(points, num_frames, body_part="All"):
	if body_part == "All":
		fig, axs = plt.subplots(7)
		for i in range(7):
			part_points = []
			for j in range(num_frames):
				part_points.append(points[j][i])

			axs[i].plot([x[0] for x in part_points], [x[1] for x in part_points], '-')
	else:
		part_points = []
		part = BODY_PARTS[body_part]
		for j in range(num_frames):
			part_points.append(points[j][part - 1])
		plt.plot([x[0] for x in part_points], [x[1] for x in part_points], '-')

	plt.show()


def plot_y(hand_y, peaks):
	# plotRightHand = list(hand.values())
	# plotRightHand = list(hand.values())
	# plotFrames = list(hand.keys())
	plt.plot(hand_y, '--')

	pos_points = []
	for peak in peaks:
		pos_points.append(hand_y[peak])

	plt.plot(peaks, pos_points, 'ro')
	plt.show()


def get_beats(num_frames):
	ret = []

	# max_y = max(right_hand.items(), key=lambda x:x[1][1])
	# max_x = max(right_hand.items(), key=lambda x:x[1][0])
	# min_x = min(right_hand.items(), key=lambda x:x[1][0])

	# Calculate and find peaks
	# RightHandY = list(right_hand.values())
	RightHandY = [x[1] for x in right_hand]
	mean = float(sum(RightHandY))/float(len(RightHandY))
	pos_peaks, prop = find_peaks(RightHandY, height=mean)

	# plot each peak as a red dot and plot y movement
	plot_y(RightHandY, pos_peaks)

	return pos_peaks

def process_data(frame_points, num_frames):
	for i in range(0, num_frames):
		right_hand.append(frame_points[i][BODY_PARTS["RWrist"] - 1])

	graph_data(frame_points, num_frames)
	return get_beats(num_frames)

points = np.load('cheetah_pose_points.npy', allow_pickle=True)
graph_data(points, 1500)
