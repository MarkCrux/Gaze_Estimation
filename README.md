# Gaze Estimation

This is a Gaze Estimation application that provides a **webcam-based gaze estimation system**. It is a simple demo solution for the Amplified Intelligence interview.


## Installation

Clone this project:

```shell
git clone https://github.com/MarkCrux/Gaze_Estimation.git
```

### For Pip install
Install these dependencies (NumPy, OpenCV, Mediapipe):

```shell
pip install -r requirements.txt
```


### Verify Installation

Run the demo:

```shell
python main.py
```

## Method

Here we list the steps for achieving the gaze estimation function.

### 1. Get face_mesh from MediaPipe.
The [MidiaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh#face-landmark-model) shows how to use the model.

This [website](https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts) provides the index of the points for the face mesh. We need this to select multiple points, including the eyes, iris, nose, chin, and etc.

### 2. Camera calibration.

Use OpenCV to calibrate my MacBook camera. This is the [tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html).

**However, in this project, the camera calibration codes were commented.** This is because you may have a different pinhole camera when you run the code. 

Also, I found that the chess board image quality has an important impact on the calibration result. Could do more research on this [paper](https://people.cs.rutgers.edu/~elgammal/classes/cs534/lectures/CameraCalibration-book-chapter.pdf) to improve the calibration.


### 3. Head pose estimation
The interview document provides a resource for [Head Pose Estimation](https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/). This function has been achieved by following the tutorial.

Head pose estimation is an important application for identifying gaze attention. However, **gaze estimation can be a more difficult task than pose estimation**. For example, when we change our head pose and move our eyes at the same time, our gaze is different from the head pose. 

### 4. Eye tracking
As mentioned in step 3, eye movement also needs to be considered when we estimate the gaze direction. 

I followed this [eye-tracking application](https://github.com/antoinelame/GazeTracking) to calculate the eye movement angle. The idea mainly includes:

#### 4.1 Select the eye points from the face mesh.
These points include the iris centre and each eye's most left and most right points.

#### 4.2 Calculate the iris position ratio.
I made the calculation by dividing the iris centre (relative position) with the eye's width.

#### 4.3 Convert the ratio to angle.
This is a rough conversion. I referred to the threshold values from [eye-tracking application](https://github.com/antoinelame/GazeTracking), which uses [0.35, 0.65]. Also, in another [head pose estimation](https://github.com/niconielsen32/ComputerVision/blob/master/headPoseEstimation.py), the threshold values for the pose angle are [-10, 10]. So I just simply mapped the range from [0.35, 0.65] to [-10, 10].

newangle = (oldangle - 0.35)/(0.65 - 0.35) * 20 - 10


### 5. Gaze estimation.
The proposed method combines the pose's angle and the eye's angle together to improve gaze estimation accuracy.


## Potential application
I found a [gaze estimation model](https://github.com/hysts/pytorch_mpiigaze_demo) on GitHub, which might be a useful reference for the interview demo. I was focusing on the interview-suggested steps, so I didn't try this project yet. The model was trained on a gaze estimation dataset, while my solution does not include any training process.

## Completeness
Minimum requirements completed, some of the desirables not completed. 

I did the task during the weekends, which is not a good time to disturb you to ask for the specs about the gaze estimation. I was mainly focusing on how to achieve gaze estimation rather than simple pose estimation. I am sorry that I didn't provide much for Docker, entrypoints, package, and UI. I am very happy to complete the rest of the task in future work.


## Licensing

This project is released by Yuchen Wei under the terms of the MIT Open Source License. View LICENSE for more information.
