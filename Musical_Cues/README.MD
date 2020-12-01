Code for running OpenPose code for human pose estimation using deep learning in OpenCV.

# On Ubuntu/Mac, run this command to download the model.
sh get_model.sh

To run this model, use the following command:
`python3 pose_video_processor.py [video]`

The output results are saved as a new image `cheetah_pose.mp4`.

To run analysis on outputed data:
`python3 main.py`
