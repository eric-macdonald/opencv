# import the necessary packages
import argparse
import datetime
from scipy import ndimage
import numpy as np
import time
import cv2
 
 # construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())


if args.get("video", None) is None:
    camera = cv2.VideoCapture(0)
    time.sleep(0.25)
else:
    camera = cv2.VideoCapture(args["video"])
				 

(grabbed, frame) = camera.read()
#fps = 30
fps = 15
height , width , layers =  frame.shape
firstFrame = None
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
##codec = cv2.VideoWriter_fourcc('I','4','2','0')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
video  = cv2.VideoWriter('output.avi',codec,fps,(width, height), True) 

num_frames = 150
start_frame = 0
frame_array = np.zeros((num_frames, height,width,3), np.uint8)

index = 0
while True:
    (grabbed, frame) = camera.read()
    text = "Unoccupied"
    if not grabbed:
        break
									 
    height , width , layers =  frame.shape
    width = width
    height = height
    dim = (width, height)
    frame = cv2.resize(frame, (width, height))
    
    # look for any changes in gray
#   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#   gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # look for changes in red
    lower1 = np.array([0, 100, 100])
    upper1 = np.array([8, 255, 255])
    lower2 = np.array([165, 100, 100])
    upper2 = np.array([179, 255, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    gray  = cv2.addWeighted(mask1, 1, mask2, 1, 0)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
														 
    if firstFrame is None:
        firstFrame = gray
 	continue

    frameDelta = cv2.absdiff(firstFrame, gray)
#   cv2.imshow("delta",frameDelta)
    ret, thresh = cv2.threshold(frameDelta, 110, 255, cv2.THRESH_BINARY)
			 
    thresh = cv2.dilate(thresh, None, iterations=2)
    image,cnts, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
							 
    for c in cnts:
        if cv2.contourArea(c) < args["min_area"]:
 	    continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"

    cv2.putText(frame, "UTEP motion detection".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    cv2.imshow("Feed", frame)
    frame_array[index] = frame
#   video.write(frame)
    print index
    index = index + 1
    if index == num_frames:
        index = 0

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

start_frame = index
while (index < num_frames): 
    video.write(frame_array[index])
    print index 
    index = index + 1

index = 0
while index < start_frame:
    video.write(frame_array[index])
    print index 
    index = index + 1

# cleanup the camera and close any open windows
camera.release()
video.release()
cv2.destroyAllWindows()
