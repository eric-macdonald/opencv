import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["video"])

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def onChange(trackbarValue):
    cap.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
    err,img = cap.read()
    cv2.imshow("mywindow", img)
    pass

cv2.namedWindow('mywindow')
cv2.createTrackbar( 'start', 'mywindow', 0, length, onChange )
cv2.createTrackbar( 'end'  , 'mywindow', 100, length, onChange )

onChange(0)
cv2.waitKey()

start = cv2.getTrackbarPos('start','mywindow')
end   = cv2.getTrackbarPos('end','mywindow')
if start >= end:
    raise Exception("start must be less than end")

cap.set(cv2.CAP_PROP_POS_FRAMES,start)
while cap.isOpened():
    err,img = cap.read()
    if cap.get(cv2.CAP_PROP_POS_FRAMES) >= end:
        break
    cv2.imshow("mywindow", img)
    k = cv2.waitKey(10) & 0xff
    if k==27:
        break



#import numpy as np
#import argparse
#import cv2

#ap = argparse.ArgumentParser()
#ap.add_argument("-v", "--video", help="path to the video file")
#ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
#args = vars(ap.parse_args())
##
#cap = cv2.VideoCapture(args["video"])
#
#while(cap.isOpened()):
#    (grabbed, frame) = cap.read()
#    if not grabbed:
#        break
#    fps = 15
#    height , width , layers =  frame.shape
#    #print height, width 
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    cv2.imshow('frame',frame)
#    if cv2.waitKey(0) & 0xFF == ord('q'):
#        break
#
#
#cap.release()
#cv2.destroyAllWindows()
