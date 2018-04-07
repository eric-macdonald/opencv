# import the necessary packages
import argparse
import datetime
import numpy as np
import time
import cv2
 
 # construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-s", "--scale", type=int, default=0.5, help="scale")
ap.add_argument("-f", "--flip", type=int, default=1, help="flip")
args = vars(ap.parse_args())
  
  # if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    quit()
else:
    camera = cv2.VideoCapture(args["video"])
scale = args["scale"]				 
flip = args["flip"]

(grabbed, frame) = camera.read()
fps = 30
height , width , layers =  frame.shape
height = int(scale * height)
width  = int(scale * width)
#fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
#codec = cv2.VideoWriter_fourcc(*'XVID') # works with play_video.py
#codec = cv2.VideoWriter_fourcc('A','V','C','1')
#codec = cv2.VideoWriter_fourcc('Y','U','V','1')
#codec = cv2.VideoWriter_fourcc('P','I','M','1')
#codec = cv2.VideoWriter_fourcc('M','J','P','G')
codec = cv2.VideoWriter_fourcc('M','P','4','2')
#codec = cv2.VideoWriter_fourcc('D','I','V','3')
#codec =  cv2.VideoWriter_fourcc('D','I','V','X')
#codec = cv2.VideoWriter_fourcc('U','2','6','3')
#codec = cv2.VideoWriter_fourcc('I','2','6','3')
#codec = cv2.VideoWriter_fourcc('F','L','V','1')
#codec = cv2.VideoWriter_fourcc('H','2','6','4')
#codec = cv2.VideoWriter_fourcc('A','Y','U','V')
#codec = cv2.VideoWriter_fourcc('I','U','Y','V')
#video  = cv2.VideoWriter('output.avi',fourcc,fps,(2*width, 2*height), True) 
videof  = cv2.VideoWriter('frame.avi',codec,fps,(width, height)) 

while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    (grabbed, frame) = camera.read()
    if not grabbed:
        break
									 
# resize the frame, convert it to grayscale, and blur it
    frame = cv2.resize(frame, (width, height))
    frame = cv2.flip(frame, flip)
    cv2.imshow("scaled", frame)
    videof.write(frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
												 
camera.release()
videof.release()
cv2.destroyAllWindows()
