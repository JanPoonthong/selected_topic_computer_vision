import cv2
import numpy as np
from scipy.signal import correlate2d, convolve2d
import matplotlib.pyplot as plt

# Load your actual grayscale image
grayscale_image = cv2.imread('/Users/janpoonthong/Pictures/jan/jan_classroom_greyscale.png', cv2.IMREAD_GRAYSCALE)

# Define the asymmetric 3x3 filter mask using digits 2, 8, 6
filter_mask = np.array([[2, 8, 6],
                        [8, 6, 2],
                        [6, 2, 8]])

# Perform correlation
correlation_result = correlate2d(grayscale_image, filter_mask, mode='same', boundary='wrap')

# Save correlation result as BMP
correlation_image_path = 'correlation_result.bmp'
cv2.imwrite(correlation_image_path, correlation_result.astype(np.uint8))

# Perform convolution
convolution_result = convolve2d(grayscale_image, filter_mask, mode='same', boundary='wrap')

# Save convolution result as BMP
convolution_image_path = 'convolution_result.bmp'
cv2.imwrite(convolution_image_path, convolution_result.astype(np.uint8))

# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(correlation_result, cmap='gray')
plt.title('Correlation Result')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(convolution_result, cmap='gray')
plt.title('Convolution Result')
plt.axis('off')

plt.show()

print(f'Correlation result saved as {correlation_image_path}')
print(f'Convolution result saved as {convolution_image_path}')
