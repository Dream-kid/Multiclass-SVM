# importing required libraries of opencv
import cv2
import  numpy as np
# importing library for plotting
from matplotlib import pyplot as plt
img = cv2.imread('img_hue hls.png')
img_to_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
img_to_yuv[:, :, 0] = cv2.equalizeHist(img_to_yuv[:, :, 0])
hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)

cv2.imwrite('result.jpg', hist_equalization_result)
# reads an input image
im = cv2.imread('result.jpg')

vals = im.mean(axis=2).flatten()
# plot histogram with 255 bins
b, bins, patches = plt.hist(vals, 255)
plt.xlim([0,255])
plt.ylim(0, 1000)
plt.show()

im = cv2.imread('img_hue hls.png')
vals = im.mean(axis=2).flatten()
# plot histogram with 255 bins
b, bins, patches = plt.hist(vals, 255)
plt.xlim([0,255])
plt.ylim(0, 1000)
plt.show()

