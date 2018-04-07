# import the necessary packages
from __future__ import print_function
import cv2
import datetime
import numpy as np
import time
import sys
 
scale = 0.4 
# load the image and convert it to grayscale
left = cv2.imread(sys.argv[1],0)
right = cv2.imread(sys.argv[2],0)
height , width = right.shape
height = int(scale*height)
width = int(scale*width)
left = cv2.resize(left, (width, height))
right = cv2.resize(right, (width, height))

# initialize the AKAZE descriptor, then detect keypoints and extract
# local invariant descriptors from the image
detector = cv2.AKAZE_create()
(kp1, des1) = detector.detectAndCompute(left, None)
(kp2, des2) = detector.detectAndCompute(right, None)
print(type(des1))
print("keypoints: {}, descriptors: {}".format(len(kp1), des1.shape))
print("keypoints: {}, descriptors: {}".format(len(kp2), des2.shape))
print(des1[0])
# draw the keypoints and show the output image
cv2.drawKeypoints(left, kp1, left, (0, 255, 0))
cv2.drawKeypoints(right, kp2, right, (0, 255, 0))
cv2.imshow("Output1", left)
cv2.waitKey(0)
cv2.imshow("Output2", right)
cv2.waitKey(0)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
both = cv2.drawMatches(left,kp1,right,kp2,matches[:10], None, flags=2)
cv2.imshow("final", both)
cv2.waitKey(0)




