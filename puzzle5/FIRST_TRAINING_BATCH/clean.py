import cv2
import numpy
import sys
import os
import struct
import scipy.misc

image = cv2.imread("abc.png")

image = cv2.bilateralFilter(image,9,75,75)

textColor = (255, 255, 255)


diffs_per_channel = image.astype(numpy.float32) - textColor

# square the distances (to get all positive numbers), and sum across the
# third dimension (i.e. sum the squared color differences for each pixel
squared_dists = (diffs_per_channel**2).sum(axis=2)

# take the square root of the result to get an actual distance
dists = numpy.sqrt(squared_dists)

# convert back to unsigned, 8-bit integers
# make all the values fit into the 0-255 range so we don't have overflow
# when we convert back into 8-bit integers
dists_display = numpy.clip(dists, 0, 255).astype(numpy.uint8)

dists_display[dists < 200] = 0
dists_display[dists > 200] = 255

# kernel = numpy.ones((1,1),numpy.uint8)
# opening = cv2.morphologyEx(dists_display, cv2.MORPH_OPEN, kernel)

kernel = numpy.ones((3,3),numpy.uint8)
closing = cv2.morphologyEx(dists_display, cv2.MORPH_CLOSE, kernel)

scipy.misc.imsave("winner.jpg", closing)
