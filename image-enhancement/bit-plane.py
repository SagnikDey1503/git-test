import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('git-test\image-enhancement\pattern_bit.png', cv2.IMREAD_GRAYSCALE)
bit_planes = []

for i in range(8):
    plane = (img >> i) & 1
    plane = plane * 255
    #bitwise shift
    bit_planes.append(plane)


plt.figure(figsize=(12, 6))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(bit_planes[7 - i], cmap='gray')  
    plt.title(f'Bit-plane {7 - i}')
    plt.axis('off')

plt.tight_layout()
plt.show()
