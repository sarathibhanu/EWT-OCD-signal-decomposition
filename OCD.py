import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import orthogonal_mp
from patchify import patchify, unpatchify

# Load grayscale image
img = cv2.imread('thumb_1.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Input image 'thumb' not found.")
img = img.astype(np.float32) / 255.0
original_shape = img.shape

# Parameters
patch_size = 8
step = 4  # Overlapping patches
dict_size = 256
sparsity = 10
max_patches = 10000
max_iter = 35
batch_size = 100

# Extract patches (with overlap)
patches = patchify(img, (patch_size, patch_size), step=step)
patches_shape = patches.shape
patches = patches.reshape(-1, patch_size, patch_size)

# Randomly sample patches for dictionary training
np.random.seed(0)
idx = np.random.choice(patches.shape[0], min(max_patches, patches.shape[0]), replace=False)
train_patches = patches[idx].reshape(len(idx), -1)

# Normalize training patches
train_patches -= np.mean(train_patches, axis=1, keepdims=True)

# Train dictionary
dict_learner = MiniBatchDictionaryLearning(
    n_components=dict_size,
    alpha=1.0,
    max_iter=max_iter,
    batch_size=batch_size,
    transform_n_nonzero_coefs=sparsity,
    transform_algorithm='omp',
    verbose=True
)
dictionary = dict_learner.fit(train_patches).components_

# Sparse coding for all patches
all_patches = patches.reshape(patches.shape[0], -1)
patch_means = np.mean(all_patches, axis=1, keepdims=True)
all_patches_centered = all_patches - patch_means

# Apply sparse coding using Orthogonal Matching Pursuit
codes = orthogonal_mp(dictionary.T, all_patches_centered.T, n_nonzero_coefs=sparsity).T

# Reconstruct patches from sparse codes
reconstructed_patches = np.dot(codes, dictionary) + patch_means
reconstructed_patches = reconstructed_patches.reshape(-1, patch_size, patch_size)

# Rebuild the image using overlapping patches
reconstructed_img = np.zeros(original_shape, dtype=np.float32)
weight = np.zeros(original_shape, dtype=np.float32)

idx = 0
for i in range(0, original_shape[0] - patch_size + 1, step):
    for j in range(0, original_shape[1] - patch_size + 1, step):
        reconstructed_img[i:i+patch_size, j:j+patch_size] += reconstructed_patches[idx]
        weight[i:i+patch_size, j:j+patch_size] += 1
        idx += 1

# Normalize by weights and clip values
reconstructed_img /= np.maximum(weight, 1e-8)
reconstructed_img = np.clip(reconstructed_img, 0, 1)

# Show original and reconstructed image
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original (Noisy) Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_img, cmap='gray')
plt.title('Reconstructed with Sparse Dictionary')
plt.axis('off')
plt.tight_layout()
plt.show()

# Save the reconstructed image
output_image = (reconstructed_img * 255).astype(np.uint8)
success = cv2.imwrite('thumb_1_ocd.jpg', output_image)
if success:
    print("Image saved successfully as 'thumb_1_ocd.jpg'")
else:
    print("Failed to save image.")