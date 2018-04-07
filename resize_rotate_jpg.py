# import the necessary packages
import argparse
import math
import datetime
import numpy as np
import time
import cv2

def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = ((width - 1) / 2.0, (height - 1) / 2.0)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

# Get Bounding Box
    radians = math.radians(angle)
    sin = abs(math.sin(radians))
    cos = abs(math.cos(radians))
    bound_w = (width * cos) + (height * sin)
    bound_h = (width * sin) + (height * cos)

# Set Translation
    rotation_mat[0, 2] += ((bound_w - 1) / 2.0 - image_center[0])
    rotation_mat[1, 2] += ((bound_h - 1) / 2.0 - image_center[1])

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (int(bound_w), int(bound_h)))
    return rotated_mat


 # construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picture", help="path to the jp file")
ap.add_argument("-s", "--scale", type=float, default=0.5, help="scale")
ap.add_argument("-r", "--rotate", type=int, default=0, help="rotate")
args = vars(ap.parse_args())
  
  # if the video argument is None, then we are reading from webcam
if args.get("picture", None) is None:
    print "need to specify jpg"
    quit()
else:
    frame = cv2.imread(args["picture"])
scale = args["scale"]				 
rotate = args["rotate"]

height , width , layers =  frame.shape
height = int(scale * height)
width  = int(scale * width)
frame = cv2.resize(frame, (width, height))
frame = rotate_image(frame, rotate)
cv2.imshow("scaled", frame)
cv2.imwrite("output.jpg", frame)
key = cv2.waitKey(0) & 0xFF
												 
cv2.destroyAllWindows()



