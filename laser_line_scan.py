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
print "file name " + str(sys.argv[1])
frame_filename = str(sys.argv[1]) + "frame.jpg"
variance_filename = str(sys.argv[1]) + "variance2.jpg"
variancex_filename = str(sys.argv[1]) + "variancex2.jpg"
variancey_filename = str(sys.argv[1]) + "variancey2.jpg"
print "frame name " + frame_filename
print "variance name " + variance_filename

minval = 1000
maxval = -1000
scan = np.ones((int(15000),int(800),1), np.float32)
frame = np.ones((int(15000),int(800),1), np.uint8)
pre_variance_frame = np.ones((int(15000),int(800),1), np.float32)
pre_variancey_frame = np.ones((int(15000),int(800),1), np.float32)
pre_variancex_frame = np.ones((int(15000),int(800),1), np.float32)
variance_frame = np.ones((int(15000),int(800),1), np.uint8)
variancex_frame = np.ones((int(15000),int(800),1), np.uint8)
variancey_frame = np.ones((int(15000),int(800),1), np.uint8)

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
print "min val = " + str(minval)
print "max val = " + str(maxval)

scaler = float((2**8) / (maxval - minval))
print scaler
adder = float(-minval)
print adder

rows    = index
columns = len(line)
print "rows and columns = " + str(rows) + " " + str(columns)
for idx1 in range(0, rows, 1):
    for idx2 in range(0, columns, 1):
        frame[idx1][idx2] = np.uint8(scaler*(scan[idx1][idx2] + adder))
        #print frame[idx1][idx2]

print("First loop = --- %s seconds ---" % (time.time() - start_time))

minvarval = 1000
maxvarval = -1000
for idx1 in range(4, rows-4, 1):
    for idx2 in range(4, columns-4, 1):
        window = []
        vert_window = []
        horz_window = []
        window = np.ones((int(5),int(5),1), np.float32)
        for offset1 in range(-2, 3, 1):
            for offset2 in range(-2, 3, 1):
                window[offset1+2][offset2+2] = frame[idx1+offset1][idx2+offset2]
       # print window
       # print window.shape
        pointvar = math.log(np.var(window)+0.000001)
       # print window[:,0]
        vert_window.append(np.average(window[:,0]))
        vert_window.append(np.average(window[:,1]))
        vert_window.append(np.average(window[:,2]))
        vert_window.append(np.average(window[:,3]))
        vert_window.append(np.average(window[:,4]))
        vert_var = math.log(np.var(vert_window)+0.000001)
        horz_window.append(np.average(window[0,:]))
        horz_window.append(np.average(window[1,:]))
        horz_window.append(np.average(window[2,:]))
        horz_window.append(np.average(window[3,:]))
        horz_window.append(np.average(window[4,:]))
        horz_var = math.log(np.var(horz_window)+0.000001)
       # print pointvar, vert_var, horz_var

        pre_variance_frame[idx1][idx2] = np.uint8(pointvar)
        if(minvarval > pointvar):
            minvarval = pointvar
        if(maxvarval < pointvar):
            maxval = pointvar
        pre_variancex_frame[idx1][idx2] = np.uint8(vert_var)
        pre_variancey_frame[idx1][idx2] = np.uint8(horz_var)
print("Second loop = --- %s seconds ---" % (time.time() - start_time))
       
scaler = float((2**8) / (maxvarval - minvarval))
print scaler
adder = -minvarval
print adder

for idx1 in range(0, rows, 1):
    for idx2 in range(0, columns, 1):
        variance_frame[idx1][idx2] = np.uint8(scaler*(pre_variance_frame[idx1][idx2] + adder))
        variancex_frame[idx1][idx2] = np.uint8(scaler*(pre_variancex_frame[idx1][idx2] + adder))
        variancey_frame[idx1][idx2] = np.uint8(scaler*(pre_variancey_frame[idx1][idx2] + adder))

print("Third loop = --- %s seconds ---" % (time.time() - start_time))

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

