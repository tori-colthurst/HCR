import sys
from video_class import video
import matplotlib.pyplot as plt
import cv2

def plot_example(orignalx, originaly, smoothx, smoothy):
    plt.scatter(orignalx, originaly, color='red')
    plt.scatter(smoothx, smoothy, color='green')
    plt.gca().invert_yaxis()
    plt.show()

'''
input - expects argument for ".npy" file to process
'''
def main(npy_file, video_file):
    chest = 0
    right_shoulder = 1
    right_elbow = 2
    right_wrist = 3
    left_shoulder = 4
    left_elbow = 5
    left_wrist = 6

    video_obj = video(npy_file, video_file)

    original_x_r_wrist = []
    original_y_r_wrist = []
    smooth_x_r_wrist = []
    smooth_y_r_wrist = []

    for frame in video_obj.frames:
        # right elbow
        if frame[right_elbow] is not None:
            r_elb = video_obj.smooth(frame[right_elbow][0], frame[right_elbow][1], right_elbow)
        # right wrist
        if frame[right_wrist] is not None:
            r_wrist = video_obj.smooth(frame[right_wrist][0], frame[right_wrist][1], right_wrist)
            original_x_r_wrist.append(frame[right_wrist][0])
            original_y_r_wrist.append(frame[right_wrist][1])
            smooth_x_r_wrist.append(r_wrist[0])
            smooth_y_r_wrist.append(r_wrist[1])

        # left elbow
        if frame[left_elbow] is not None:
            l_elb = video_obj.smooth(frame[left_elbow][0], frame[left_elbow][1], left_elbow)
        # left wrist
        if frame[left_wrist] is not None:
            l_wrist = video_obj.smooth(frame[left_wrist][0], frame[left_wrist][1], left_wrist)

        video_obj.draw_frame()

    cv2.destroyAllWindows()

    # plot_example(original_x_r_wrist, original_y_r_wrist, smooth_x_r_wrist, smooth_y_r_wrist)

if __name__ == "__main__":

    npy_file = "cheetah_pose_points.npy"
    video_file = "cheetah_pose.mp4"
    main(npy_file, video_file)
