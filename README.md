# Gaze_Estimation

This is a Gaze Estimation application that provides a **webcam-based eye tracking system**. It is a simple demo solution for Amplified Intelligence interview.

## Installation

Clone this project:

```shell
git clone https://github.com/antoinelame/GazeTracking.git
```

### For Pip install
Install these dependencies (NumPy, OpenCV, Dlib):

```shell
pip install -r requirements.txt
```

> The Dlib library has four primary prerequisites: Boost, Boost.Python, CMake and X11/XQuartx. If you doesn't have them, you can [read this article](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/) to know how to easily install them.


### For Anaconda install
Install these dependencies (NumPy, OpenCV, Dlib):

```shell
conda env create --file environment.yml
#After creating environment, activate it
conda activate GazeTracking
