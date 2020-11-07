Code for running OpenPose code for human pose estimation using deep learning in OpenCV.

First step is download the pretrained Coco model:

# On Ubuntu/Mac, run this command to download the model.
sh get_model.sh

To run this model, use the following command:
`python3 pose_estimation.py --input sample.jpg`

The output results are saved as a new image `result_sample.jpg`.
