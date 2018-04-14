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

short_name = sys.argv[1]
length = sys.argv[2]
print "start of analyze.py"
print "file name " + str(sys.argv[1])
frame_filename = str(sys.argv[1]) + "frame.jpg"
variance_filename = str(sys.argv[1]) + "variance.jpg"
variancex_filename = str(sys.argv[1]) + "variancex.jpg"
variancey_filename = str(sys.argv[1]) + "variancey.jpg"
print "frame name " + frame_filename
print "variance name " + variance_filename

minval = 1000
maxval = -1000
scan = np.ones((int(length),int(800),1), np.float32)
frame = np.ones((int(length),int(800),1), np.uint8)
pre_variance_frame = np.ones((int(length),int(800),1), np.float32)
pre_variancey_frame = np.ones((int(length),int(800),1), np.float32)
pre_variancex_frame = np.ones((int(length),int(800),1), np.float32)
variance_frame = np.ones((int(length),int(800),1), np.uint8)
variancex_frame = np.ones((int(length),int(800),1), np.uint8)
variancey_frame = np.ones((int(length),int(800),1), np.uint8)

file = open(sys.argv[1], 'r') 
index = 0 
means = []
sum = 0
for idx1, line in enumerate(file):
    line = line.split(",")
    line = np.asarray(line, dtype=np.float32)
    index = index + 1
    for idx2, value in enumerate(line):
        if(minval > value) and (value > -5):
            minval = value
        if(maxval < value):
            maxval = value
        scan[idx1][idx2] = np.float32(value)
        if((idx1%400) == 0):
            if((idx2%400) == 0):
                print scan[idx1][idx2], idx1, idx2
            
print("Finished reading in raw data = --- %s seconds ---" % (time.time() - start_time))
print "min val = " + str(minval)
print "max val = " + str(maxval)

scaler = float((2**8) / (maxval - minval))
adder = float(-minval)

print "scaler for scan"
print adder,scaler

rows    = index
columns = len(line)
print "rows and columns = " + str(rows) + " " + str(columns)
for idx1 in range(0, rows, 1):
    for idx2 in range(0, columns, 1):
        temp = float(scaler*(scan[idx1][idx2] + adder))
        frame[idx1][idx2] = np.uint8(temp)
        if((idx1%400) == 0):
            if((idx2%400) == 0):
                print frame[idx1][idx2], idx1, idx2


print("Finished converting raw data = --- %s seconds ---" % (time.time() - start_time))

minvarval = 1000
maxvarval = -1000
minvarvalx = 1000
maxvarvalx = -1000
minvarvaly = 1000
maxvarvaly = -1000
for idx1 in range(4, rows-4, 1):
    for idx2 in range(4, columns-4, 1):
        window = []
        vert_window = []
        horz_window = []
        window = np.ones((int(5),int(5),1), np.float32)
        vert_window = np.ones((int(5),1), np.float32)
        horz_window = np.ones((int(5),1), np.float32)
        for offset1 in range(-2, 3, 1):
            for offset2 in range(-2, 3, 1):
                window[offset1+2][offset2+2] = scan[idx1+offset1][idx2+offset2]
        pointvar = math.log(np.var(window) + 0.0000001)
        #pointvar = (np.var(window) + 0.0000001)
        vert_window[0] = np.average(window[:,0])
        vert_window[1] = np.average(window[:,1])
        vert_window[2] = np.average(window[:,2])
        vert_window[3] = np.average(window[:,3])
        vert_window[4] = np.average(window[:,4])
        vert_var = math.log(np.var(vert_window) + 0.0000001)
        horz_window[0] = np.average(window[0,:])
        horz_window[1] = np.average(window[1,:])
        horz_window[2] = np.average(window[2,:])
        horz_window[3] = np.average(window[3,:])
        horz_window[4] = np.average(window[4,:])
        horz_var = math.log(np.var(horz_window) + 0.0000001)
#        print pointvar, idx1, idx2
        if((idx1%400) == 0):
            if((idx2%400) == 0):
#                print("Inside second loop = --- %s seconds ---" % (time.time() - start_time))
                print pointvar, vert_var, horz_var, idx1, idx2

        pre_variance_frame[idx1][idx2] = pointvar
        if(minvarval > pointvar):
            minvarval = pointvar
        if(maxvarval < pointvar):
            maxvarval = pointvar

        pre_variancex_frame[idx1][idx2] = vert_var
        if(minvarvalx > vert_var):
            minvarvalx = vert_var
        if(maxvarvalx < vert_var):
            maxvarvalx = vert_var

        pre_variancey_frame[idx1][idx2] = horz_var
        if(minvarvaly > horz_var):
            minvarvaly = horz_var
        if(maxvarvaly < horz_var):
            maxvarvaly = horz_var
            
print("finished variance analysis = --- %s seconds ---" % (time.time() - start_time))
       
scaler = float((2**8) / (maxvarval - minvarval))
adder = -minvarval
scalerx = float((2**8) / (maxvarvalx - minvarvalx))
adderx = -minvarvalx
scalery = float((2**8) / (maxvarvaly - minvarvaly))
addery = -minvarvaly
print "scaling"
print maxvarval, minvarval
print scaler, adder
print "scalingX"
print maxvarvalx, minvarvalx
print scalerx, adderx
print "scalingY"
print maxvarvaly, minvarvaly
print scalery, addery

#print "pointvar, vert_var, horz_var, tempvar, tempvarx, tempvary, variance_frame[idx1][idx2], variancex_frame[idx1][idx2], variancey_frame[idx1][idx2], idx1, idx2"
for idx1 in range(0, rows, 1):
    for idx2 in range(0, columns, 1):
        tempvar =  scaler*(pre_variance_frame[idx1][idx2] + adder)
        tempvarx = scalerx*(pre_variancex_frame[idx1][idx2] + adderx)
        tempvary = scalery*(pre_variancey_frame[idx1][idx2] + addery)
        variance_frame[idx1][idx2] = np.uint8(tempvar)
        variancex_frame[idx1][idx2] = np.uint8(tempvarx)
        variancey_frame[idx1][idx2] = np.uint8(tempvary)
        if((idx1%400) == 0):
            if((idx2%400) == 0):
                print pointvar, vert_var, horz_var, tempvar, tempvarx, tempvary, variance_frame[idx1][idx2], variancex_frame[idx1][idx2], variancey_frame[idx1][idx2], idx1, idx2


print("finished variance scaling = --- %s seconds ---" % (time.time() - start_time))

frame          = cv2.equalizeHist(frame)
frame          = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
variance_frame = cv2.equalizeHist(variance_frame)
variance_frame = cv2.applyColorMap(variance_frame, cv2.COLORMAP_JET)
variancex_frame = cv2.equalizeHist(variancex_frame)
variancex_frame = cv2.applyColorMap(variancex_frame, cv2.COLORMAP_JET)
variancey_frame = cv2.equalizeHist(variancey_frame)
variancey_frame = cv2.applyColorMap(variancey_frame, cv2.COLORMAP_JET)

cv2.imwrite(frame_filename, frame)
cv2.imwrite(variance_filename, variance_frame)
cv2.imwrite(variancex_filename, variancex_frame)
cv2.imwrite(variancey_filename, variancey_frame)

print("writing out jpg files = --- %s seconds ---" % (time.time() - start_time))
