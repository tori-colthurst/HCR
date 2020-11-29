import numpy as np
import cv2
import argparse
import process_results as pr

'''
read_video()
input: video_file_path - relative path to video
output: video_frames - list of frames from video
'''
def read_video(video_file_path, num_frames):
	cap = cv2.VideoCapture(video_file_path)
	if not cap.isOpened() :
		print("ERROR: failed to open video file")
		return -1

	# read video frame by frame
	video_frames = []
	if num_frames:
		i = 0
		while cap.isOpened() and i<num_frames:
			ret, frame = cap.read()
			i += 1
			if ret:
				# display frame
				cv2.imshow("Frame", frame)

				# # press Q on keyboard to exit
				if cv2.waitKey(40) & 0xFF == ord('q'): # 40 ms = 25 fps
					break

				video_frames.append(frame)
			else:
				print("Nothing read. Exiting video read loop")
				break
	else:
		while cap.isOpened():
			ret, frame = cap.read()
			if ret:
				video_frames.append(frame)
			else:
				print("Nothing read. Exiting video read loop")
				break

	# release video capture object and close all frames
	cap.release()
	cv2.destroyAllWindows()
	return video_frames


def main():
	parser = argparse.ArgumentParser(description='Program to run pose recognition on a video file')

	parser.add_argument('video_path', help='Relative path to video to be processed')
	parser.add_argument('--num_frames', default=0, type=float, help='Number of frames to process')
	parser.add_argument('--thr', default=0.1, type=float, help='Threshold value for pose parts heat map')
	parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
	parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

	args = parser.parse_args()

	video_frames = read_video(args.video_path, args.num_frames)

	if video_frames != -1:
		# loop processing here
		BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
					   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
					   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
					   "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

		POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
					   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"] ]

		inWidth = args.width
		inHeight = args.height

		net = cv2.dnn.readNetFromCaffe("pose/coco/deploy_coco.prototxt", "pose/coco/pose_iter_440000.caffemodel")

		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		out_vid = cv2.VideoWriter(args.video_path+'_pose.mp4', fourcc, 24.0, (video_frames[0].shape[1],video_frames[0].shape[0]))

		frames_points = []
		print("Num Frames: {}".format(len(video_frames)))
		frame_num = 0
		for frame in video_frames:
			print("Frame Num: {}".format(frame_num))
			frame_num += 1
			frameWidth = frame.shape[1]
			frameHeight = frame.shape[0]

			inp = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
										  (0, 0, 0), swapRB=False, crop=False)
			net.setInput(inp)
			out = net.forward()
			points = []
			for i in range(len(BODY_PARTS)):
				# Slice heatmap of corresponging body's part.
				heatMap = out[0, i, :, :]

				# Originally, we try to find all the local maximums. To simplify a sample
				# we just find a global one. However only a single pose at the same time
				# could be detected this way.
				_, conf, _, point = cv2.minMaxLoc(heatMap)
				x = (frameWidth * point[0]) / out.shape[3]
				y = (frameHeight * point[1]) / out.shape[2]

				# Add a point if it's confidence is higher than threshold.
				points.append((int(x), int(y)) if conf > args.thr else None)
			frames_points.append(points[BODY_PARTS["Neck"]:BODY_PARTS["LWrist"]+1])

			for pair in POSE_PAIRS:
				partFrom = pair[0]
				partTo = pair[1]
				assert(partFrom in BODY_PARTS)
				assert(partTo in BODY_PARTS)

				idFrom = BODY_PARTS[partFrom]
				idTo = BODY_PARTS[partTo]
				if points[idFrom] and points[idTo]:
					cv2.line(frame, points[idFrom], points[idTo], (255, 74, 0), 3)
					cv2.ellipse(frame, points[idFrom], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
					cv2.ellipse(frame, points[idTo], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
					cv2.putText(frame, str(idFrom), points[idFrom], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)
					cv2.putText(frame, str(idTo), points[idTo], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)
			# out_vid.write(frame)

		frames_points_np = np.array(frames_points)
		peaks = pr.process_data(frames_points_np, frame_num)
		for i in range(frame_num):
			frame = video_frames[i]
			if peaks[i] == 1:
				cv2.putText(frame, "Beat", (int(frameWidth / 2), frameHeight - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)
			else:
				cv2.putText(frame, "No Beat", (int(frameWidth / 2), frameHeight - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)
			out_vid.write(frame)

		out_vid.release()

		with open(args.video_path+'_points.npy', 'wb') as f:
			np.save(f, frames_points_np)


if __name__ == "__main__":
	main()
