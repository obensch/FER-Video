# Facial Emotion Recognition
Facial Emotion Recognition in a video stream using a pre-trained Deep-CNN model.

The video stream is resized and limited to about 5FPS to boost performance and reduce delays.

The model used was trained using this repository: https://github.com/obensch/fer-2013
## requirements
Python version: 3.7

The following python packages are required
* numpy
* openCV
* imutils
* time
* tensorflow/keras

## change settings in the scripts
Webcam is enabled by default in all scripts. 
To enable webstream comment the line:
```python
cam = cv2.VideoCapture(0)
```
and uncomment the line
```python
# cam = cv2.VideoCapture("http://192.168.178.21:8080/video?type=some.mjpg")
```
To change the frame settings edit the folling lines in all the scritps:
```python
# set frame size
FrameWidth = 1280
FrameHeight = 720
```

## Script: main.py 
Requirements:
The files 'haarcascade_frontalface_default.xml' is required for face detection.
The file can be downloaded e.g. from here: https://github.com/opencv/opencv/tree/master/data/haarcascades

This script can be executed using the command:
```python
python main.py
```

### Change the loaded model
The model can be changed in line 19:
```python
model.load_weights('main.h5')
```
