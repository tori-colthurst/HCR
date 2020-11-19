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
    frame_count = 0
    while (cap.isOpened()) or frame_count < 100:
        ret, frame = cap.read()
        if ret == True:
            # # display frame
            # cv2.imshow("Frame", frame)
            #
            # # press Q on keyboard to exit
            # if cv2.waitKey(100) & 0xFF == ord('q'):
            #     break
            video_frames.append(frame)

        else:
            print("Nothing read. Exiting read video loop.")
            break
        frame_count += 1

    # release video capture object and close all frames
    cap.release()
    cv2.destroyAllWindows()

    return video_frames

def main():
    read_video("Footage/Theo_540_fb.mov")

if __name__ == "__main__":
    main()
