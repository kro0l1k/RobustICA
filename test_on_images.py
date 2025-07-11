import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.decomposition import FastICA
from skimage import data, img_as_float
from skimage.color import rgb2gray
from skimage.transform import resize

# Add the parent directory to the path so we can import from RICA_code
from RICA import Optimizer

# 1. Load / prepare sources (use any two equal-sized images)
img1 = rgb2gray(img_as_float(data.astronaut()))
img2 = img_as_float(data.camera())
# Make them 256×256 for speed
img1 = resize(img1, (256, 256), anti_aliasing=True)
img2 = resize(img2, (256, 256), anti_aliasing=True)

# 2. Vectorise: each image is one column
S = np.stack((img1.ravel(), img2.ravel()), axis=1)  # shape (65536, 2)

print("Shape of source signals:", S.shape)
n, d = S.shape  # n: number of pixels, d: number of sources

# 3. Artificial mixing matrix 
A = np.array([[0.6, 0.4],
              [0.4, 0.6]])
X = S @ A.T                                       # mixed signals

# 4. FastICA
print("Shape of X:", X.shape)
ica = FastICA(n_components=2, whiten='unit-variance', random_state=0)
S_hat = ica.fit_transform(X)     # separated
print("S shape after FastICA:", S_hat.shape)
components = S_hat.T.reshape((2, 256, 256))       # back to image shape

# 5. RICA:
rica_optimizer = Optimizer(n, d, 5)

I_x = rica_optimizer.RICA(X)

S_hat_rica = X @ I_x.T              # separated

# 5. Visualise
fig, axes = plt.subplots(2, 4, figsize=(8, 5))
axes = axes.ravel()
axes[0].imshow(img1, cmap='gray'); axes[0].set_title('Source 1')
axes[1].imshow(img2, cmap='gray'); axes[1].set_title('Source 2')
axes[2].imshow((X[:,0].reshape(256,256)), cmap='gray'); axes[2].set_title('Mix 1')
axes[3].imshow((X[:,1].reshape(256,256)), cmap='gray'); axes[3].set_title('Mix 2')
axes[4].imshow(components[0], cmap='gray'); axes[4].set_title('ICA comp 1')
axes[5].imshow(components[1], cmap='gray'); axes[5].set_title('ICA comp 2')
axes[6].imshow(S_hat_rica[:,0].reshape(256,256), cmap='gray'); axes[6].set_title('RICA comp 1')
axes[7].imshow(S_hat_rica[:,1].reshape(256,256), cmap='gray'); axes[7].set_title('RICA comp 2')
for ax in axes: ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()
