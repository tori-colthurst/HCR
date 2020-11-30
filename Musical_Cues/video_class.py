import numpy as np
import cv2
import argparse


class video:

    def __init__(self, npy_file_name, video_file_name):

        # points.shape = (9561,7)
        # where each 9561 is each frame, 7 is the 7 points
        # each point is an (x,y) tuple with (0,0) in the top left corner
        # x increasing to the right, y increasing down
        self.frames = np.load(npy_file_name, allow_pickle=True)
        self.num_points = 7

        self.picture_frames = self.read_video(video_file_name)

        # x values, y values for all 7 points
        # self.running_avg[0] = [[x_list], [y_list]] for 1st point
        self.running_avg = np.zeros((self.num_points,2))
        self.running_std = np.zeros((self.num_points,2))

        self.window_size = 50
        self.window = np.zeros((self.num_points, 2, self.window_size))
        self.window.fill(-1) # initialize window values to -1
        self.window_idx = 0

        self.N = 0
        self.first = 10 # are we within the first few frames?

    def update_avg(self, x, y, point):
        i = self.window_idx
        self.window[point][0][i] = x
        self.window[point][1][i] = y
        print((x,y))
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
        print("New average: ("+str(avg_x)+", "+str(avg_y)+")")
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
        print("New std: ("+str(std_x**0.5)+", "+str(std_y**0.5)+")")

        self.N += 1
        self.window_idx = (self.window_idx + 1) % self.window_size

    def smooth(self, x, y, point):
        keep_x = x
        keep_y = y

        avg_x = self.running_avg[point][0]
        avg_y = self.running_avg[point][1]
        std_x = self.running_std[point][0]
        std_y = self.running_std[point][1]

        if std_x != 0.0 and (x > avg_x + 4*std_x or x < avg_x - 4*std_x):
            keep_x = avg_x # outlier in x
        if std_y != 0.0 and (y > avg_y + 3*std_y or y < avg_y - 3*std_y):
            keep_y = avg_y # outlier in x

        self.update_avg(keep_x, keep_y, point)
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
