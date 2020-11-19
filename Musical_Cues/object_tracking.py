from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
import cv2
import video_processing as vp

'''
 function
description:
inputs:
outputs: -1, error
'''
def object_tracking(frames_list):

    # Discriminative Correlation Filter (with Channel and Spatial Reliability).
    tracker = cv2.TrackerCSRT_create()
    # initialize bounding box coordinates and fps for object
    BB_coord = None
    fps = None

    for frame in frames_list:

        # resize frame to process faster
        frame = imutils.resize(frame, width=500)
        (H, W) = frame.shape[:2]

        # currently tracking object, not first frame
        if BB_coord is not None:

            # grab new bounding box coordinates of object
            success, box = tracker.update(frame)

            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            fps.update()
            fps.stop()

            # write over frame
            text = "Success" if success else "Failure"
            cv2.putText(frame, text, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # display image
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(10) & 0xFF

        # use 'S' key to DRAW bounding box to track with mouse
        if key == ord("s"):
            # select bounding box
            BB_coord = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
            # start tracker and fps
            tracker.init(frame, BB_coord)
            fps = FPS().start()
        # quit
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

def main():
    frames_list = vp.read_video("Footage/Theo_540_fb.mov")
    object_tracking(frames_list)

if __name__ == "__main__":
    main()
