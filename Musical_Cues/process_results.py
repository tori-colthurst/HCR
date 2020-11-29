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

	# Calculate and find peaks - both negative and positive
	RightHandX = [x[0] for x in right_hand]
	RightHandY = [x[1] for x in right_hand]
	mean = float(sum(RightHandX))/float(len(RightHandX))
	x_pos_peaks, prop = find_peaks(RightHandX, height=mean)
	mean = float(sum(RightHandY))/float(len(RightHandY))
	y_pos_peaks, prop = find_peaks(RightHandY, height=mean)

	RightHandXNeg = [-1 * x[0] for x in right_hand]
	RightHandYNeg = [-1 * x[1] for x in right_hand]
	mean = float(sum(RightHandXNeg))/float(len(RightHandXNeg))
	x_neg_peaks, prop = find_peaks(RightHandXNeg, height=mean)
	mean = float(sum(RightHandYNeg))/float(len(RightHandYNeg))
	y_neg_peaks, prop = find_peaks(RightHandYNeg, height=mean)

	x_neg_peaks = list(x_neg_peaks)
	y_neg_peaks = list(y_neg_peaks)
	x_pos_peaks = list(x_pos_peaks)
	y_pos_peaks = list(y_pos_peaks)
	all_peaks = x_pos_peaks + x_neg_peaks + y_pos_peaks + y_neg_peaks
	# plot each peak as a red dot and plot y movement
	# plot_y(RightHandY, y_pos_peaks + y_neg_peaks)
	# plot_y(RightHandX, x_pos_peaks + x_neg_peaks)

	# Convert to num_frames size array
	# 1 = beat, 0 = no beat
	for i in range(num_frames):
		if i in all_peaks:
			ret.append(1)
		else:
			ret.append(0)

	return ret

def process_data(frame_points, num_frames):
	for i in range(0, num_frames):
		right_hand.append(frame_points[i][BODY_PARTS["RWrist"] - 1])

	# graph_data(frame_points, num_frames)
	return get_beats(num_frames)

if __name__ == "__main__":
	num_frames = 100
	points = np.load('cheetah_pose_points.npy', allow_pickle=True)
	for i in range(0, 100):
		right_hand.append(points[i][BODY_PARTS["RWrist"] - 1])
	# graph_data(points, 100)
	x = get_beats(100)
