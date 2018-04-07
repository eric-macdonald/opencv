import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["video"])

#while(cap.isOpened()):
#    ret, frame = cap.read()
#    cv2.imshow('frame',frame)
#    cv2.waitKey(0)

while(cap.isOpened()):
    (grabbed, frame) = cap.read()
    if not grabbed:
        break
    fps = 15
    height , width , layers =  frame.shape
    #print height, width 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',frame)
#    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
