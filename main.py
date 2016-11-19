import cv2
import numpy as np
from matplotlib import pyplot as plt

filename = 'manga.jpg'

orig = cv2.imread(filename)
img = cv2.imread(filename)
result = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY_INV)

contours, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    hull = cv2.convexHull(cnt)
    cv2.drawContours(result, [cnt], -1, 255, -1)
    cv2.drawContours(result, [hull], -1, 255, -1)

plt.subplot(121), plt.imshow(orig)
plt.subplot(122), plt.imshow(result)
plt.show()
