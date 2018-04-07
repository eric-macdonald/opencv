import numpy as np
import collections 
import sys
import time
from euclid import *
import datetime
import re
import cv2
import math
import datetime
import time

def find_spatter(contours, frame):    
    max_area = 0
    mass = 0
    found_pool = False
    centroid = (0,0)
    for cnt_index in range(0, len(contours)):
        c = contours[cnt_index]
        area = float(cv2.contourArea(c))
        if (max_area < area):
            max_area = area
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centroid = (cx,cy)
    if(max_area > 300):
        found_pool = True
        print "found left centroid " + str(centroid) + " " + str(max_area)
        cv2.putText(frame, str(centroid), centroid, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
        cv2.circle(frame, centroid, 1, (255, 0, 255), 5)

    for cnt_index in range(0, len(contours)):
        c = contours[cnt_index]
        M = cv2.moments(c)
        cx = 0
        cy = 0
        area = float(cv2.contourArea(c))
        area_in = max_area > area and min_area < area 
        if(area_in):
            x,y,w,h = cv2.boundingRect(c)    
            bmx = x + int(w/2)
            bmy = y + h
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            top = float(math.hypot(float(box[1][0] - box[0][0]), float(box[1][1] - box[0][1])))
            side = float(math.hypot(float(box[2][0] - box[1][0]), float(box[2][1] - box[1][1])))
            if(top > side) and (side != 0):
                aspect = float(top/side)
                thin = side
                rect_area = top*side 
                extent = float(area)/rect_area
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                solidity = float(area)/hull_area
                solidity_thresh = 1 - float(1/aspect) - 0.2
                mid1 = (int((box[2][0] + box[1][0])/2), int((box[2][1] + box[1][1])/2))
                mid2 = (int((box[3][0] + box[0][0])/2), int((box[3][1] + box[0][1])/2))
            elif(side != 0):
                aspect = float(side/top) 
                thin = top
                rect_area = top*side 
                extent = float(area)/rect_area
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                solidity = float(area)/hull_area
                solidity_thresh = 1 - float(1/aspect) - 0.2
                mid1 = (int((box[1][0] + box[0][0])/2), int((box[1][1] + box[0][1])/2))
                mid2 = (int((box[3][0] + box[2][0])/2), int((box[3][1] + box[2][1])/2))

            if(area_in and ((aspect > 5) or ((aspect > 3.5) and solidity > solidity_thresh))):
                mass = mass + area
                if(inLine(mid1,mid2,centroid) or not (found_pool) or True): # bypassing this condition
                    cv2.drawContours(frame, [c], 0, (0, 255, 0), 1)
                    cv2.circle(frame, mid1, 1, (0, 0, 255), 5)
                    cv2.circle(frame, mid2, 1, (0, 0, 255), 5)
                    cv2.putText(frame, str(mid1), mid1, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(frame, str(mid2), mid2, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            else:
                bob = 0

    text_string1 = "Mass = "
    text_string2 = str(int(mass))
    text_string3 = "Youngstown State University"
    cv2.putText(frame, text_string1, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(frame, text_string2, (180, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(frame, text_string3, (10, 680), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    return frame

def inLine(a,b,c):
    crossproduct = (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1])
    if abs(crossproduct) > 100:
        print "out of line" 
        print -1*crossproduct
        return False
    else:
        print "in of line" 
        print crossproduct
        return True
    
    
    
    
print "import done"
##########################################
#parameters and initializing variables
frame = np.ones((int(720),int(2560),3), np.uint8)
framenum = 0
scale = 1.0 
warp = False
canny1 = 10
canny2 = 50
min_length = scale*10 
min_area   = 40 
max_area   = 5000 
minyl = 50 
maxyl = 400 
minxl= 150
maxxl= 800 
minyr = 50 
maxyr = 400 
minxr= 150
maxxr= 600 
lowerHSV = np.array([0, 0, 80])
upperHSV = np.array([180, 255, 255])

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

##########################################
# Input arguments
##########################################
if(len(sys.argv) <  2):
    print("Usage: python SLS.py <left.file> <right.file> ")
    quit()
elif(len(sys.argv) ==  3):
    print "using video"
    color = "black" 
    cameraL = cv2.VideoCapture(sys.argv[1])
    cameraR = cv2.VideoCapture(sys.argv[2])
    time.sleep(1.00)
    grabbed, frame = cameraL.read()
    if(grabbed is None):
        print("nothing grabbed")
    else:
        print "grabbedL" + str(grabbed)
        print "frameL" + str(frame.shape)
    grabbed, frame = cameraR.read()
    if(grabbed is None):
        print("nothing grabbed")
    else:
        print "grabbedR" + str(grabbed)
        print "frameR" + str(frame.shape)
else:
    print("Usage: python spatter.py <left.file> <right.file>")
    quit()


print "camera and output initializaiton"

firstFrame = True
fps = 15
width = 1280
height = 720
double_output_size = (2*width, height)
output_size = (width, height)

codec = cv2.VideoWriter_fourcc('M','J','P','G')
fps = 15
videof = cv2.VideoWriter()
videod = cv2.VideoWriter()
success = videod.open('sls.avi',codec,fps,double_output_size,True) 
success = videof.open('single.avi',codec,fps,output_size,True) 

if(success):
    print "video initialization successful"
frame_index = 0
horizontal_count = 0
print "starting loop"
keep = [] 
while(1):
    retL,frameL= cameraL.read()
    retR,frameR= cameraR.read()
    if(retL != True) or (retR != True):
        videof.release()
        cv2.destroyAllWindows()
        cameraL.release()
        cameraR.release()
	quit()
    print "grabbed two frames"

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    retL,maskL = cv2.threshold(grayL,90,255,cv2.THRESH_BINARY)
    retR,maskR = cv2.threshold(grayR,90,255,cv2.THRESH_BINARY)

    if(firstFrame):
         old_dilated_maskL = cv2.dilate(maskL,None,iterations = 0)
         old_dilated_maskR = cv2.dilate(maskR,None,iterations = 0)
         firstFrame = False
    else:
         old_dilated_maskL = dilated_maskL
         old_dilated_maskR = dilated_maskR

    dilated_maskL = cv2.dilate(maskL,None,iterations = 0)
    dilated_maskR = cv2.dilate(maskR,None,iterations = 0)
    flip_oldL = cv2.bitwise_not(old_dilated_maskL)
    flip_oldR = cv2.bitwise_not(old_dilated_maskR)
    flip_oldL = cv2.erode(flip_oldL, None, iterations = 0)
    flip_oldR = cv2.erode(flip_oldR, None, iterations = 0)
    #showmask = cv2.resize(flip_oldL, (640,320))
    #cv2.imshow('flip old', showmask)
    maskL = cv2.bitwise_xor(old_dilated_maskL, dilated_maskL)
    maskR = cv2.bitwise_xor(old_dilated_maskR, dilated_maskR)
    #showmask = cv2.resize(maskL, (640,320))
    #cv2.imshow('XOR', showmask)
    maskL = cv2.erode(maskL,None,iterations = 0)
    maskR = cv2.erode(maskR,None,iterations = 0)
    maskL = cv2.bitwise_and(maskL, flip_oldL)
    maskR = cv2.bitwise_and(maskR, flip_oldR)

    (imgL, cntsL, hierarchyL) = cv2.findContours(maskL.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    (imgR, cntsR, hierarchyR) = cv2.findContours(maskR.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    framenum = framenum + 1
    frameL = find_spatter(cntsL,frameL)
    frameR = find_spatter(cntsR,frameR)

    frame = np.concatenate((frameL, frameR), axis=1)
    frameshow = cv2.resize(frame, (1280, 360))
    videod.write(frame)
    videof.write(frameL)
    cv2.imshow('frameshow', frameshow)
    cv2.waitKey(0)
    if cv2.waitKey(33) == ord('a'):
        print "you pressed a"
        break
videof.release()
cv2.destroyAllWindows()
