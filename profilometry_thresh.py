#!/Users/eric.w.macdonald/miniconda2/bin/python
import numpy as np
import statistics
from scipy.signal import lfilter, lfilter_zi, filtfilt, butter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import datetime
import sys
import cv2
import math
import time
start_time = time.time()
def nothing(x):
    pass

def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

scale = 0.2

cv2.namedWindow('image')

cv2.createTrackbar('threshold','image',0,255,nothing)
u = 255 
l = 0

short_name = sys.argv[1]
print "file name " + str(sys.argv[1])
out_filename = str(sys.argv[1]) + "thresh.jpg"
print "output name " + out_filename
frame = cv2.imread(short_name)
height, width, layers = frame.shape
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

resized = (int(width*scale), int(height*scale))
codec = cv2.VideoWriter_fourcc('M','J','P','G')
fps = 5 
videof = cv2.VideoWriter()
success = videof.open('thresh.avi',codec,fps,resized,True)

for index in range(0,255,1):
    ret, thresh = cv2.threshold(gray, index, 255, cv2.THRESH_BINARY)
    thresh2      = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)
    cv2.putText(thresh2, str(index), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 6.0, (0, 0, 255), 4)
    cv2.putText(thresh2, str(index), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 6.0, (0, 0, 255), 4)
    thresh2 = cv2.resize(thresh2, resized)
    thresh = cv2.resize(thresh, resized)
    filename = short_name + str(index) + ".jpg"
    cv2.imwrite(filename, thresh)
    videof.write(thresh2)

#while (1):
#    k = cv2.waitKey(0)
#    if k==27:    # Esc key to stop
#        break
#    else:
#        print u, l # else print its value
#    index=cv2.getTrackbarPos('threshold','image')
#    ret, thresh = cv2.threshold(gray, index, 255, cv2.THRESH_BINARY)
#    thresh      = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)
#    cv2.putText(thresh, str(index), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 6.0, (0, 0, 255), 4)
#    thresh = cv2.resize(thresh, resized)
#    cv2.imshow('image',thresh)
#    videof.write(thresh)
     
