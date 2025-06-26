import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data




img = data.camera()
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

total_pixels = img.shape[0] * img.shape[1]
hist_normalized = hist / total_pixels

plt.figure(figsize=(8, 4))
plt.plot(hist_normalized, color='black')
plt.title('Normalized Histogram')
plt.xlabel('Pixel Intensity (0–255)')
plt.ylabel('Frequency (Normalized)')
plt.grid(True)
plt.tight_layout()
plt.show()