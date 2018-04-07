import cv2
import sys
import argparse
import datetime
import numpy as np
import time
import cv2

# Get user supplied values
cascPath = sys.argv[1]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

camera = cv2.VideoCapture(0)
time.sleep(0.5)

# Read the image
#image = cv2.imread(imagePath)

firstFrame = None
(grabbed, frame) = camera.read()
#fps = camera.get(cv2.CAP_PROP_FPS)
fps = 30
print "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)
print "fps" + str(fps)
height , width , layers =  frame.shape
codec = cv2.VideoWriter_fourcc('I','4','2','0')
video  = cv2.VideoWriter('faces.avi',codec,fps,(width, height)) 

while True:
    (grabbed, frame) = camera.read()
    if not grabbed:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if firstFrame is None:
        firstFrame = gray
 	continue

# Detect faces in the image
    faces = faceCascade.detectMultiScale(
         gray,
         scaleFactor=1.1,
         minNeighbors=5,
         minSize=(30, 30),
         flags = cv2.CASCADE_SCALE_IMAGE
         )

    print "Found {0} faces!".format(len(faces))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    video.write(frame)
    cv2.imshow("Faces found", frame)

    key = cv2.waitKey(1) & 0xFF
   # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

camera.release()
video.release()
cv2.destroyAllWindows()
