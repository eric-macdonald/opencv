import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

#img = cv2.imread('juliet1.jpg',0)
#img = cv2.imread('canny.jpg',0)
input_file = sys.argv[1]
img = cv2.imread(input_file,0)
img2 = cv2.imread(input_file)
#cv2.imshow('sand',img2)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 40*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
#plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#disp_color = cv2.applyColorMap( disparity_visual, cv2.COLORMAP_JET)
plt.show()
