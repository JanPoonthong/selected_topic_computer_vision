#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:41:33 2024

@author: janpoonthong

1. Perform direct Fourier transform onto the grayscale image. Apply a low-pass spectral filter.
2. Perform inverse Fourier transform onto the modified spectrum and compare the original and modified images.

"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read the color image and convert it to grayscale
image = cv2.imread('/Users/janpoonthong/University/Year-3/Semester-1/selected_topic_in_computer_vision/quiz_2/desk.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform direct Fourier transform
f = np.fft.fft2(gray_image)
fshift = np.fft.fftshift(f)

# Apply a low-pass spectral filter
rows, cols = gray_image.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols), np.uint8)
r = 50  # Radius of the low-pass filter
center = (crow, ccol)
x, y = np.ogrid[:rows, :cols]
mask_area = (x - crow)**2 + (y - ccol)**2 <= r*r
mask[mask_area] = 1

fshift = fshift * mask

# Perform inverse Fourier transform
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Display the images
plt.figure(figsize=(10, 10))
plt.subplot(131), plt.imshow(gray_image, cmap='gray'), plt.title('Original Image')
plt.subplot(132), plt.imshow(np.log(np.abs(fshift)), cmap='gray'), plt.title('Fourier Transform')
plt.subplot(133), plt.imshow(img_back, cmap='gray'), plt.title('Filtered Image')
plt.show()
