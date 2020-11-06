import numpy as np
import cv2

'''
read_video function
description:
inputs:
outputs: -1, error
'''
def read_video(video_file_path):

    cap = cv2.VideoCapture(video_file_path)

    if (cap.isOpened()== False):
        print("ERROR: failed to open video file.")
        return -1

    # read video frame by frame
    video_frames = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # display frame
            cv2.imshow("Frame", frame)

            # press Q on keyboard to exit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            # video_frames.append(frame)
        else:
            print("Nothing read. Exiting read video loop.")
            break

    # release video capture object and close all frames
    cap.release()
    cv2.destroyAllWindows()

def main():
    read_video("Footage/Theo_540_w_Audio.mp4")

if __name__ == "__main__":
    main()