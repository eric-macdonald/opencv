import cv2
import sys
import argparse
import datetime
import numpy as np
import time
import cv2

# Get user supplied values
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-c", "--cascPath",  help="cascade")
args = vars(ap.parse_args())

# Create the haar cascade
Cascade = cv2.CascadeClassifier(args["cascPath"])

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    camera = cv2.VideoCapture(0)
    time.sleep(0.1)
#    quit()
else:
    camera = cv2.VideoCapture(args["video"])

(grabbed, frame) = camera.read()
fps = 30
codec = cv2.VideoWriter_fourcc('M','P','4','2')
height , width , layers =  frame.shape
videof  = cv2.VideoWriter('frame.avi',codec,fps,(width, height)) 

firstFrame = None
print "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)
print "fps" + str(fps)
height , width , layers =  frame.shape
codec = cv2.VideoWriter_fourcc('I','4','2','0')
videof  = cv2.VideoWriter('features.avi',codec,fps,(width, height)) 

while True:
    (grabbed, frame) = camera.read()
    if not grabbed:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if firstFrame is None:
        firstFrame = gray
 	continue

# Detect features in the image
    features = Cascade.detectMultiScale(
         gray,
         scaleFactor=1.1,
         minNeighbors=5,
         minSize=(30, 30),
         flags = cv2.CASCADE_SCALE_IMAGE
         )

    print "Found {0} features!".format(len(features))

    for (x, y, w, h) in features:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Faces found", frame)
    videof.write(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

camera.release()
videof.release()
cv2.destroyAllWindows()
