# import the necessary packages
import argparse
import datetime
import numpy as np
import time
import cv2
 
 # construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())
  
  # if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
#    camera = cv2.VideoCapture(args["video"])
    camera = cv2.VideoCapture('bullet.mov')
#   camera = cv2.VideoCapture(0)
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
#fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
#codec = cv2.VideoWriter_fourcc(*'XVID') # works with play_video.py
codec = cv2.VideoWriter_fourcc('I','4','2','0')
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
videof  = cv2.VideoWriter('frame.avi',codec,fps,(2*width, 2*height)) 
videog = cv2.VideoWriter('gray.avi',codec,fps,(2*width, 2*height), 0) 
videot = cv2.VideoWriter('thresh.avi',codec,fps,(2*width, 2*height), 0) 

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
    width = 2*width
    height = 2*height
    #print width, height, dim
    frame = cv2.resize(frame, (width, height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    gray = cv2.medianBlur(gray, 31)
														 
# if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        lastFrame = gray
 	continue


    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
#    frameDelta = cv2.absdiff(lastFrame, gray)

    ret, thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)
    #ret, thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)
			 
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
#    thresh = cv2.dilate(thresh, None, iterations=3)
    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
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
        cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 255, 255), 2)

#    cv2.putText(frame, "high speed camera".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    #height , width , layers =  frame.shape
    #print height, width, layers
    cv2.imshow("Threshold", frame)
#   cv2.imshow("Security Feed", frame)
    videof.write(frame)
    videog.write(gray)
    videot.write(thresh)
#    cv2.imshow("Thresh", thresh)
#    cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF
   # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
												 
# cleanup the camera and close any open windows
camera.release()
videof.release()
videog.release()
videot.release()
cv2.destroyAllWindows()
