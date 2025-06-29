import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_ubyte


image = img_as_ubyte(data.camera()) 


mean = 0
stddev = 25
gaussian_noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)
noisy_image = cv2.add(image.astype(np.float32), gaussian_noise)
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

avg_3x3 = cv2.blur(noisy_image, (3, 3))
avg_5x5 = cv2.blur(noisy_image, (5, 5))
avg_7x7 = cv2.blur(noisy_image, (7, 7))


titles = ['Original', 'Noisy', '3x3 Avg Filter', '5x5 Avg Filter', '7x7 Avg Filter']
images = [image, noisy_image, avg_3x3, avg_5x5, avg_7x7]

plt.figure(figsize=(15, 6))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
