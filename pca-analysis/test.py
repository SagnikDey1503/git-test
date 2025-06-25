import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
import os

image_path = os.path.join('git-test', 'pca-analysis', 'hello.png')
image = io.imread(image_path)

if image.ndim == 3 and image.shape[-1] == 4:
    image = color.rgba2rgb(image)

gray_image = color.rgb2gray(image) if image.ndim == 3 else image
gray_image = resize(gray_image, (128, 128), anti_aliasing=True)

mu = np.mean(gray_image, axis=0)
X_centered = gray_image - mu

cov = np.cov(X_centered, rowvar=False)
eig_vals, eig_vecs = np.linalg.eigh(cov)

idx = np.argsort(eig_vals)[::-1]
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:, idx]

k = 20
U_k = eig_vecs[:, :k]
Y_k = X_centered @ U_k
X_recon = (Y_k @ U_k.T) + mu

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Original Grayscale Image")
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Reconstructed (k={k})")
plt.imshow(X_recon, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
