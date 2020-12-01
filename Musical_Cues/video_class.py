import numpy as np
import cv2
import argparse


class video:

	def __init__(self, npy_file_name, video_file_name):
		self.chest = 0
		self.right_shoulder = 1
		self.right_elbow = 2
		self.right_wrist = 3
		self.left_shoulder = 4
		self.left_elbow = 5
		self.left_wrist = 6

		# points.shape = (9561,7)
		# where each 9561 is each frame, 7 is the 7 points
		# each point is an (x,y) tuple with (0,0) in the top left corner
		# x increasing to the right, y increasing down
		self.frames = np.load(npy_file_name, allow_pickle=True)
		self.new_frames = np.zeros((9561,7,2))
		self.num_points = 7

		self.picture_frames = self.read_video(video_file_name)
		self.draw_idx = 0

		# x values, y values for all 7 points
		# self.running_avg[0] = [[x_list], [y_list]] for 1st point
		self.running_avg = np.zeros((self.num_points,2))
		self.running_std = np.zeros((self.num_points,2))

		self.window_size = 24
		self.window = np.zeros((self.num_points, 2, self.window_size))
		self.window.fill(-1) # initialize window values to -1
		self.window_idx = 0

		self.N = 0
		self.first = 10 # are we within the first few frames?

		self.velocity_window = 12
		self.left_wrist_x_v = 0
		self.left_wrist_y_v = 0

		self.right_wrist_x_v = 0
		self.right_wrist_y_v = 1
		# self.beat_count = 0
		self.last_beat = 0
		self.tempo = 0
		self.fps = 24
		self.num_sec = 3
		self.curr_num_sec = 0
		self.beat_window = np.zeros(self.num_sec)
		self.beat_window.fill(-1)
		self.beat_idx = 0
		# self.heavy_window = 24*1
		# self.articulation_status = [0]*self.heavy_window
		# self.articulation_idx = 0
		self.trap_window_size = 24*1
		self.trap_window = np.zeros(self.trap_window_size)
		self.trap_idx = 0

		self.Ltrap_window_size = 24*3
		self.Ltrap_window = np.zeros(self.Ltrap_window_size)
		self.Ltrap_idx = 0

	def update_avg(self, x, y, point):
		i = self.window_idx
		self.window[point][0][i] = x
		self.window[point][1][i] = y
		# print((x,y))
		# print(self.window[point])

		# calculate window average
		avg_x = 0
		avg_y = 0
		count = 0
		for j in range(0, self.window_size):
			xt = self.window[point][0][j]
			yt = self.window[point][1][j]
			if xt != -1:
				avg_x += xt
				avg_y += yt
				count += 1
		avg_x /= count
		avg_y /= count
		# print("New average: ("+str(avg_x)+", "+str(avg_y)+")")
		self.running_avg[point][0] = avg_x
		self.running_avg[point][1] = avg_y

		# calculate window std deviation
		std_x = 0
		std_y = 0
		for j in range(0, self.window_size):
			xt = self.window[point][0][j]
			yt = self.window[point][1][j]
			if xt != -1:
				std_x += (xt - avg_x)**2
				std_y += (yt - avg_y)**2
		std_x /= count
		std_y /= count
		self.running_std[point][0] = std_x**0.5
		self.running_std[point][1] = std_y**0.5
		# print("New std: ("+str(std_x**0.5)+", "+str(std_y**0.5)+")")

		self.N += 1
		self.window_idx = (self.window_idx + 1) % self.window_size

	def smooth(self, x, y, point):
		keep_x = x
		keep_y = y

		avg_x = self.running_avg[point][0]
		avg_y = self.running_avg[point][1]
		std_x = self.running_std[point][0]
		std_y = self.running_std[point][1]

		if avg_x > 0 and (x > avg_x + 150 or x < avg_x - 150):
			keep_x = avg_x # outlier in x
		if avg_y > 0 and (y > avg_y + 200 or y < avg_y - 200):
			keep_y = avg_y # outlier in x
		# if avg_x > 0:
		#     if std_x < 1 and (x > avg_x + 200 or x < avg_x - 200):
		#         keep_x = avg_x # outlier in x
		#     elif x > avg_x + 4*std_x or x < avg_x - 4*std_x:
		#         keep_x = avg_x # outlier in x
		# if avg_y > 0:
		#     if std_y < 1 and (avg_y + 150 or y < avg_y - 150):
		#         keep_y = avg_y # outlier in x
		#     elif y > avg_y + 3*std_y or y < avg_y - 3*std_y:
		#         keep_y = avg_y # outlier in x

		self.update_avg(keep_x, keep_y, point)
		self.new_frames[self.draw_idx][point][0] = int(keep_x)
		self.new_frames[self.draw_idx][point][1] = int(keep_y)
		return (keep_x, keep_y)

	def read_video(self, video_file_path):
		num_frames = self.frames.shape[0]

		cap = cv2.VideoCapture(video_file_path)
		if not cap.isOpened() :
			print("ERROR: failed to open video file")
			return -1

		# read video frame by frame
		video_frames = []
		i = 0
		while cap.isOpened() and i<num_frames:
			ret, frame = cap.read()
			i += 1
			if ret:
				video_frames.append(frame)
			else:
				print("Nothing read. Exiting video read loop")
				break

		# release video capture object and close all frames
		cap.release()
		cv2.destroyAllWindows()
		return video_frames

	def draw_frame(self):
		i = self.draw_idx
		img = self.picture_frames[i]
		font = cv2.FONT_HERSHEY_SIMPLEX

		# draw right wrist
		tup = (int(self.new_frames[i][self.right_wrist][0]), int(self.new_frames[i][self.right_wrist][1]))
		cv2.circle(img, tup, 10, (0,255,0), -1)
		# draw left wrist
		tup = (int(self.new_frames[i][self.left_wrist][0]), int(self.new_frames[i][self.left_wrist][1]))
		cv2.circle(img, tup, 10, (0,0,255), -1)

		# draw right elbow
		tup = (int(self.new_frames[i][self.right_elbow][0]), int(self.new_frames[i][self.right_elbow][1]))
		cv2.circle(img, tup, 10, (0,255,0), -1)
		# draw left elbow
		tup = (int(self.new_frames[i][self.left_elbow][0]), int(self.new_frames[i][self.left_elbow][1]))
		cv2.circle(img, tup, 10, (0,0,255), -1)

		# if self.draw_idx > 7190:
		cres = self.de_crescendo()
		art = self.articulation()
		beat = self.beats()

		if cres == -1:
			cv2.putText(img, "volume: decrescendo", (10,20), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
		elif cres == 1:
			cv2.putText(img, "volume: crescendo", (10,20), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
		else:
			cv2.putText(img, "volume: ", (10,20), font, 0.5, (0,255,0), 1, cv2.LINE_AA)

		if art == 0:
			cv2.putText(img, "style: lighter", (10,40), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
		elif art == 1:
			cv2.putText(img, "style: heavier", (10,40), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
		else:
			cv2.putText(img, "style: steady", (10,40), font, 0.5, (0,255,0), 1, cv2.LINE_AA)

		if beat == 1:
			cv2.putText(img, "Beat", (10,60), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
		elif beat == 0:
			cv2.putText(img, "No Beat", (10,60), font, 0.5, (0,255,0), 1, cv2.LINE_AA)

		self.tempo_calc()
		cv2.putText(img, str(self.tempo), (10,80), font, 0.5, (0,255,0), 1, cv2.LINE_AA)

		if self.draw_idx < self.last_beat + 3:
			cv2.circle(img, (92, 70), 10, (0,255,0), -1)
		else:
			cv2.circle(img, (92, 70), 10, (0,0,255), -1)

		# if self.frames[i] is not None and self.frames[i][self.right_wrist] is not None:
		#     if abs(self.frames[i][self.right_wrist][1] - self.new_frames[i][self.right_wrist][1]) > 50:
		# if i % 24 == 0:
		cv2.imshow('image', img)
		cv2.waitKey(40)
		# cv2.destroyAllWindows()

		self.draw_idx += 1

	def velocity(self):
		num = self.velocity_window - 1

		if self.draw_idx >= num:
			upper = self.new_frames[self.draw_idx][self.left_wrist][0]
			lower = self.new_frames[self.draw_idx-num][self.left_wrist][0]
			self.left_wrist_x_v = (upper - lower) / self.velocity_window

			upper = self.new_frames[self.draw_idx][self.left_wrist][1]
			lower = self.new_frames[self.draw_idx-num][self.left_wrist][1]
			self.left_wrist_y_v = (upper - lower) / self.velocity_window

			return (self.left_wrist_x_v, self.left_wrist_y_v)
		else:
			return (0, 0)

	def avg_y_change(self):
		avg = 0
		for i in range(0, self.velocity_window):
			avg += abs(self.new_frames[self.draw_idx-i][self.left_wrist][1] - \
					self.new_frames[self.draw_idx-i][self.right_wrist][1])
		avg /= self.velocity_window
		return avg

	def de_crescendo(self):
		v_x, v_y = self.velocity()

		if abs(v_x) < 1 and self.avg_y_change() > 10:
			if v_y > 2: # decrescendo
				return -1
			elif v_y < -2: # crescendo
				return 1
		else:
			return 0

	def trapezoid_area(self):
		lw = (self.new_frames[self.draw_idx][self.left_wrist][0], self.new_frames[self.draw_idx][self.left_wrist][1])
		le = (self.new_frames[self.draw_idx][self.left_elbow][0], self.new_frames[self.draw_idx][self.left_elbow][1])
		rw = (self.new_frames[self.draw_idx][self.right_wrist][0], self.new_frames[self.draw_idx][self.right_wrist][1])
		re = (self.new_frames[self.draw_idx][self.right_elbow][0], self.new_frames[self.draw_idx][self.right_elbow][1])

		a = abs(lw[0] - rw[0])
		b = abs(le[0] - re[0])
		h = (abs(lw[1]-le[1]) + abs(rw[1]-re[1])) / 2
		area = (a+b)*h/2

		self.trap_window[self.trap_idx] = area
		avg = 0
		for val in self.trap_window:
			avg += val
		avg /= self.trap_window_size
		self.trap_idx = (self.trap_idx+1) % self.trap_window_size

		self.Ltrap_window[self.Ltrap_idx] = area
		Lavg = 0
		for val in self.Ltrap_window:
			Lavg += val
		Lavg /= self.Ltrap_window_size
		self.Ltrap_idx = (self.Ltrap_idx+1) % self.Ltrap_window_size

		return avg, Lavg

	# def shoulder_check(self):
	#     ls = (self.new_frames[self.draw_idx][self.left_shoulder][0], self.new_frames[self.draw_idx][self.left_shoulder][1])
	#     rs = (self.new_frames[self.draw_idx][self.right_shoulder][0], self.new_frames[self.draw_idx][self.right_shoulder][1])
	#     lw = (self.new_frames[self.draw_idx][self.left_wrist][0], self.new_frames[self.draw_idx][self.left_wrist][1])
	#     rw = (self.new_frames[self.draw_idx][self.right_wrist][0], self.new_frames[self.draw_idx][self.right_wrist][1])
	#
	#     return (lw[0] < ls[0] and rw[0] > rs[0])

	def articulation(self):

		# shoulder_check = self.shoulder_check()
		avg, Lavg = self.trapezoid_area()
		# print("Area: "+str(area))

		if avg < Lavg-4000:
			# self.articulation_status[self.articulation_idx] = 0
			return 0 # lighter
		elif avg > Lavg+2500:
			# self.articulation_status[self.articulation_idx] = 1
			return 1 # heavier
		else:
			return -1
		# else:
		#     # self.articulation_status[self.articulation_idx] = 2
		#     return 2 # heavy

		# if 2 in self.articulation_status: # if heavy in last second
		#     return 2
		# else:
		#     return self.articulation_status[self.articulation_idx]

		# self.articulation_idx = (self.articulation_idx + 1) % self.heavy_window

	def velocity_right(self):
		delta = 2

		if self.draw_idx >= delta:
			upper = self.new_frames[self.draw_idx][self.right_wrist][0]
			lower = self.new_frames[self.draw_idx-delta+1][self.right_wrist][0]
			extra_low = self.new_frames[self.draw_idx-delta*2+1][self.right_wrist][0]
			curr_vel_x = (upper - lower) / (delta)
			prev_vel_x = (lower - extra_low) / delta*2

			upper = self.new_frames[self.draw_idx][self.right_wrist][1]
			lower = self.new_frames[self.draw_idx-delta+1][self.right_wrist][1]
			extra_low = self.new_frames[self.draw_idx-delta*2+1][self.right_wrist][1]
			curr_vel_y = (upper - lower) / (delta)
			prev_vel_y = (lower - extra_low) / delta*2

			return curr_vel_x, curr_vel_y, prev_vel_x, prev_vel_y
		else:
			return 0, 0, 0, 0

	def beats(self):
		x_change = False
		y_change = False
		v_x, v_y, pv_x, pv_y = self.velocity_right()
		# print(v_x)

		if self.draw_idx - self.last_beat <= 7:
			return 0

		tempo = 1
		# print((v_x, v_y))

		# if v_x < 0 and self.right_wrist_x_v > 0:
		if v_x < -tempo and pv_x > tempo:
			x_change = True
			# self.right_wrist_x_v = v_x
		# elif v_x > 0 and self.right_wrist_x_v < 0:
		elif v_x > tempo and pv_x < -tempo:
			x_change = True
			# self.right_wrist_x_v = v_x

		if v_y < -tempo and pv_y > tempo:
			y_change = True
			# self.right_wrist_y_v = v_y
		elif v_y > tempo and pv_y < -tempo:
			y_change = True
			# self.right_wrist_y_v = v_y

		if x_change or y_change:
			# if v_x == 0:
			# 	self.right_wrist_x_v = v_x
			# if v_y == 0:
			# 	self.right_wrist_y_v = v_y
			self.last_beat = self.draw_idx
			# self.beat_count += 1
			if self.beat_window[self.beat_idx] == -1:
				self.beat_window[self.beat_idx] = 1
			else:
				self.beat_window[self.beat_idx] += 1
			return 1

		return 0

	def tempo_calc(self):
		count = 0
		beat_count = 0
		for i in range(0, self.num_sec):
			if self.beat_window[i] > -1:
				count += 1
				beat_count += self.beat_window[i]
		if count > 0 and self.draw_idx % self.fps == 0:
			# print(self.beat_count)
			self.tempo = int(beat_count * 60/count)
			# self.beat_count = 0
		if self.draw_idx % self.fps == 0:
			print(self.beat_window)
			self.beat_idx = (self.beat_idx+1)%self.num_sec
			self.beat_window[self.beat_idx] = -1

##### calculate running average
# prev_avg_x = self.running_avg[point][0]
# prev_avg_y = self.running_avg[point][1]
#
# avg_x = (prev_avg_x * N + x) / (N + 1)
# avg_y = (prev_avg_y * N + y) / (N + 1)
#
# self.running_avg[point][0] = avg_x
# self.running_avg[point][1] = avg_y
#
# std_x = self.running_std[point][0]**2 * (N-1)/N + (x-prev_avg_x)**2 / (N+1)
# std_y = self.running_std[point][1]**2 * (N-1)/N + (y-prev_avg_y)**2 / (N+1)
#
# self.running_std[point][0] = std_x
# self.running_std[point][1] = std_y
