# Face detection inspired from: https://github.com/shantnu/Webcam-Face-Detect/blob/master/webcam.py
import numpy as np
import cv2 
import sys
import time
import tensorflow as tf 
import imutils

# set frame size
FrameWidth = 1280
FrameHeight = 720

# # Get webcam
cam = cv2.VideoCapture(0)
# or get Video Stream
# cam = cv2.VideoCapture("http://192.168.178.21:8080/video?type=some.mjpg")

# load model from second assignment
model = tf.keras.models.load_model('model.h5')

# Create the haar cascade
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

# labels for prediction
labels = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

# time since last rendered frame and start time
startTime = time.time()
prevTime = startTime   

while True:
    # Capture frame-by-frame
    ret, frame = cam.read()

    # get current time
    curTime = time.time()

    # elapsed since last frame
    elapsed = curTime - prevTime

    # if more than 0.2 seconds elapsed, render new frame ~ 5 frames per second
    if(elapsed > 0.2):
        # reset timer for new frame
        prevTime = curTime

        # resize for performance and convert to gray
        smallFrame = imutils.resize(frame, width=FrameWidth, height=FrameHeight)
        gray = cv2.cvtColor(smallFrame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = faceCascade.detectMultiScale(gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(40, 40),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        # transform frame for emotion detection
        cropped = []
        for i in range(0, len(faces)):
            cropped.append(frame.copy())
            x, y, w, h = faces[i]
            # change crop image size for better detection
            cropped[i] = cropped[i][y:y+h+50, x-25:x+w+25] 
            cropped[i] = cv2.resize(cropped[i], (48,48))
            cropped[i] = cv2.cvtColor(cropped[i], cv2.COLOR_BGR2GRAY)


        # run through all detected faces
        predictions = []
        for i in range(0, len(faces)):
            # predict emotions using the loaded model
            prediction_classes = model.predict(cropped[i].reshape(1,48,48,1))
            prediction = np.argmax(prediction_classes, axis=-1)
            predictions.append(labels[prediction[0]])
            # label the detected emotion
            label = labels[prediction[0]]

            x, y, w, h = faces[i]
            # draw rectangle around found face
            cv2.rectangle(smallFrame, (x-25, y), (x+w+25, y+h+50), (0, 255, 0), 2) #  (x-25, y), (x+w+25, y+h+50)
            
            # draw detected emotion
            cv2.putText(smallFrame, label.upper(), (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255))

        # show image
        cv2.imshow("Live FER using Deep-CNNs", smallFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything is done, release the capture
cam.release()
cv2.destroyAllWindows()