# import the necessary packages
import argparse
import datetime
import numpy as np
#import imutils
import time
import cv2
 
 # construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())
  
  # if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
#    camera = cv2.VideoCapture('bullet.avi')
    camera = cv2.VideoCapture(0)
    time.sleep(0.25)
			 
 # otherwise, we are reading from a video file
else:
    camera = cv2.VideoCapture(args["video"])
				 
 # initialize the first frame in the video stream
firstFrame = None
# loop over the frames of the video
(grabbed, frame) = camera.read()
fps = 15
height , width , layers =  frame.shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video  = cv2.VideoWriter('output.mov',fourcc,fps,(width, height), True) 

while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    (grabbed, frame) = camera.read()
    text = "Unoccupied"
    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        break
									 
# resize the frame, convert it to grayscale, and blur it
    height , width , layers =  frame.shape
    width = width
    height = height
    dim = (width, height)
    print width, height, dim
    frame = cv2.resize(frame, (width, height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
														 
# if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
 	continue


    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    ret, thresh = cv2.threshold(frameDelta, 70, 255, cv2.THRESH_BINARY)
    #ret, thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)
			 
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    image,cnts, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
							 
    # loop over the contours
    for c in cnts:
	# if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
 	    continue
		 
# compute the bounding box for the contour, draw it on the frame,
# and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"

	# draw the text and timestamp on the frame
    cv2.putText(frame, "high speed camera".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

	# show the frame and record if the user presses a key
#    height , width , layers =  frame.shape
#    print height, width, layers
    cv2.imshow("Security Feed", frame)
    video.write(frame)
#    cv2.imshow("Thresh", thresh)
#    cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF
   # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
												 
# cleanup the camera and close any open windows
camera.release()
video.release()
cv2.destroyAllWindows()
