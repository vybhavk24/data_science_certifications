# DL_c4_m1

## Computer Vision

### 1. Direct Definition

Computer vision is the field of deep learning and artificial intelligence that enables machines to interpret and understand visual data—images or videos—by extracting meaningful information such as object identities, positions, and actions.

### 2. Concept Intuition

Computer vision treats an image as a grid of pixel values. Unlike tabular data, nearby pixels are highly correlated: edges, textures, and shapes emerge from local neighborhoods. Convolutional neural networks (CNNs) exploit this locality, scanning small patches across the entire image to learn spatial hierarchies of features—from simple edges in early layers to complex object parts in deeper layers.

Why it matters:

- Visual data is everywhere: medical scans, autonomous driving, surveillance, and social media.
- Manual feature engineering (SIFT, HOG) was labor-intensive; CNNs learn features directly from raw pixels.
- Modern breakthroughs (ImageNet, self-driving cars) hinge on deep vision models.

### 3. Mathematical Breakdown

### 3.1 Image as a Tensor

An RGB image of height H and width W is represented as

```python
X ∈ ℝ^{H×W×3}
```

where each pixel $X[i,j,c]∈[0,255]$ or [0,1] after normalization.

### 3.2 Normalization

Scale pixels to zero mean, unit variance per channel:

```python
μ = mean(X, axis=(0,1))
σ = std(X,  axis=(0,1))
X_norm = (X - μ) / σ
```

This speeds up convergence by stabilizing the data distribution.

### 3.3 Convolution (Preview)

Sliding a filter $W∈ℝ^{f×f×3}$ over X produces a feature map Z:

```python
Z[i,j] = sum_over(f×f×3)( W * X_region ) + b
```

### 4. Code & Practical Application

Below is a minimal pipeline to load, preprocess, and display an image using Python, NumPy, and Matplotlib.

```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 1. Load image and convert to NumPy
img = Image.open('cat.jpg')                     # replace with your path
X   = np.array(img, dtype=np.float32)           # shape (H, W, 3)

# 2. Inspect shape and dtype
print("Shape:", X.shape, "Dtype:", X.dtype)

# 3. Normalize per channel
mu = X.mean(axis=(0,1), keepdims=True)
sd = X.std (axis=(0,1), keepdims=True)
X_norm = (X - mu) / (sd + 1e-8)

# 4. Display
plt.imshow((X_norm - X_norm.min()) / (X_norm.max() - X_norm.min()))
plt.title("Normalized Image")
plt.axis('off')
plt.show()
```

This pipeline is the starting point for any vision model: loading, normalizing, and batching images.

### 5. Visualization / Geometry

Visualizing pixel grids:

- **Pixel grid**: Each cell is a pixel; intensity/color values form patterns.
- **Local receptive fields**: Convolutional filters only “see” a small window (e.g., 3×3), allowing the network to detect edges or textures.
- **Feature hierarchy**: Stacking layers grows the receptive field. Early layers detect lines; deeper layers detect corners, parts, and full objects.

```
Image (H×W) → Conv 3×3 → Edge Map → Conv 5×5 → Texture Map → … → Object Class
```

### 6. Common Pitfalls & Tips

- Forgetting to normalize RGB channels separately leads to slow or unstable training.
- Mixing channel order (H×W×3 vs. 3×H×W) between libraries causes shape mismatches.
- Loading large images without resizing leads to out-of-memory errors on GPU.
- Ignoring data augmentation (flips, rotations) limits the model’s ability to generalize.

### 7. Interview-Ready Insights

- Explain why convolution is more efficient than a fully connected layer on images: parameter sharing & sparse connectivity.
- Discuss the difference between convolution and cross-correlation (most DL frameworks implement cross-correlation).
- Be ready to compare classical CV features (SIFT/HOG) vs. learned CNN filters.
- Know common architectures (LeNet, AlexNet, VGG, ResNet) and what innovations they introduced.

### 8. Practice Exercises

1. Load any RGB image, convert it to grayscale by averaging channels, then normalize and display both versions side by side.
    - Hint: $grayscale = X[...,0]0.2989 + X[...,1]0.5870 + X[...,2]*0.1140$
2. Implement a simple edge detector:
    - Define a 3×3 Sobel X filter:
        
        ```python
        sobel_x = np.array([[+1, 0, -1],
                            [+2, 0, -2],
                            [+1, 0, -1]], dtype=np.float32)
        ```
        
    - Convolve it over a grayscale image using nested loops (no external conv functions) and visualize the edge map.
3. Using TensorFlow or PyTorch, build a single-layer CNN that takes 32×32×3 images and outputs 10 scores.
    - Print the model summary.
    - Verify that the number of parameters equals (f×f×3×num_filters + num_filters).

---

## Edge Detection Example

---

### 1. Direct Definition

Edge detection identifies points in an image where intensity changes sharply. These “edges” often correspond to object boundaries, texture shifts, or lighting changes.

### 2. Concept Intuition

Edges are like the outlines of shapes in a picture. When you move from a light region to a dark one, the pixel values change abruptly. Detecting these transitions isolates contours without caring about interior texture. In deep learning, early convolutional filters learn to act as edge detectors—finding horizontal, vertical, and diagonal boundaries—which form the building blocks for more complex feature extraction.

### 3. Mathematical Breakdown

### 3.1 Image Gradient

An image (I) is a function $(I(x,y))$. The gradient measures change in intensity:

```python
∂I/∂x = I(x+1, y) - I(x-1, y)
∂I/∂y = I(x, y+1) - I(x, y-1)
```

We combine these into a gradient magnitude:

```python
G(x,y) = sqrt((∂I/∂x)**2 + (∂I/∂y)**2)
θ(x,y) = arctan2(∂I/∂y, ∂I/∂x)
```

### 3.2 Sobel Operator

Sobel filters approximate these derivatives with smoothing:

```python
Sobel_x = [[+1, 0, -1],
           [+2, 0, -2],
           [+1, 0, -1]]

Sobel_y = [[+1, +2, +1],
           [ 0,  0,  0],
           [-1, -2, -1]]
```

Convolution with these kernels yields (\partial I/\partial x) and (\partial I/\partial y).

### 4. Code & Practical Application

### 4.1 Manual Convolution with NumPy

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load and grayscale
img = Image.open('cat.jpg').convert('L')
I   = np.array(img, dtype=np.float32)

# Define Sobel kernels
sobel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)

# Convolution function
def convolve2d(image, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh//2, kw//2
    padded = np.pad(image, ((pad_h,pad_h),(pad_w,pad_w)), mode='reflect')
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            out[i,j] = np.sum(region * kernel)
    return out

# Compute gradients
Gx = convolve2d(I, sobel_x)
Gy = convolve2d(I, sobel_y)

# Gradient magnitude
G = np.hypot(Gx, Gy)
G = (G / G.max()) * 255

# Display results
plt.figure(figsize=(12,4))
for idx, (mat, title) in enumerate([(I, 'Original'),
                                    (Gx, 'Gradient X'),
                                    (Gy, 'Gradient Y'),
                                    (G, 'Magnitude')]):
    plt.subplot(1,4,idx+1)
    plt.imshow(mat, cmap='gray')
    plt.title(title)
    plt.axis('off')
plt.show()
```

### 4.2 Using PyTorch’s Conv2d

```python
import torch
import torch.nn.functional as F

# Prepare tensor [1,1,H,W]
tensor = torch.from_numpy(I).unsqueeze(0).unsqueeze(0)

# Create Sobel kernels
kernel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
kernel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32)

kernels = torch.stack([kernel_x, kernel_y]).unsqueeze(1)  # shape [2,1,3,3]

# Convolve
grad = F.conv2d(tensor, kernels, padding=1)
Gx_t, Gy_t = grad[0]
G_t = torch.sqrt(Gx_t**2 + Gy_t**2)
```

### 5. Visualization / Geometry

- **Gradient X** highlights vertical edges; bright where intensity changes horizontally.
- **Gradient Y** highlights horizontal edges; bright where intensity changes vertically.
- **Magnitude** combines both, showing all edges regardless of orientation.

Geometrically, you’re computing directional derivatives over small patches: the Sobel filter blends smoothing (kernel weights) with differencing.

### 6. Common Pitfalls & Tips

- Failing to convert to grayscale: color channels must be reduced before using single-channel kernels.
- Ignoring padding: without proper padding, edges near image borders vanish.
- Not normalizing output: raw gradient values can overflow image-display ranges.
- Using too large kernel for tiny features: small kernels (3×3) capture fine details; larger ones blur edges.

### 7. Interview-Ready Insights

- Explain why Sobel includes smoothing (weights of 2) vs. simple finite differences.
- Compare Sobel to Prewitt and Scharr: trade-off between rotational symmetry and noise sensitivity.
- Discuss Canny edge detector steps: Gaussian blur, gradient, non-maximum suppression, double thresholding, edge tracking.
- Highlight that CNNs learn edge-detecting filters automatically in early layers, often resembling Sobel-like patterns.

### 8. Practice Exercises

1. **Implement Prewitt Filter**
    - $Prewitt_x = [[1,0,-1],[1,0,-1],[1,0,-1]]$
    - Compare Prewitt magnitude to Sobel; visualize both side by side.
2. **Thresholding Edge Map**
    - Binarize gradient magnitude by selecting a threshold (T).
    - Experiment with global vs. Otsu’s method to choose (T).
3. **Build a Conv Layer with Sobel Initialization**
    - In PyTorch, create `nn.Conv2d(1,2,3,padding=1,bias=False)` and set its weights to Sobel_x and Sobel_y.
    - Forward an image batch and verify you recover the manual gradients.
4. **Edge-Aware Blurring**
    - Use gradient magnitude to create a mask: blur only regions with low gradient (smooth areas), preserve edges sharp.
    - Apply Gaussian blur modulated by this mask.

---

## Advanced Edge Detection Deep Dive

### 1. Direct Definition

Advanced edge detection encompasses algorithms that not only compute gradients but also suppress noise, enforce thin contours, and apply adaptive thresholding to produce crisp, reliable edges. The canonical example is the Canny edge detector, which combines smoothing, gradient estimation, non-maximum suppression, and hysteresis thresholding.

### 2. Concept Intuition

While simple filters (Sobel, Prewitt, Scharr) highlight intensity changes, they also amplify noise and produce thick edges. Advanced detectors aim to:

- Smooth out noise before computing gradients
- Precisely localize edges by thinning wide responses
- Connect broken segments via dual thresholds
- Adapt to varying contrast within the same image

This pipeline yields edges that are both accurate and robust—critical for tasks like object segmentation, stereo matching, and feature extraction in vision pipelines.

### 3. Mathematical Breakdown

### 3.1 Gaussian Smoothing

Before gradient, blur with a Gaussian

```python
G(x,y) = (1 / (2πσ²)) * exp(-(x² + y²) / (2σ²))
```

Convolve image I with G to get I_smooth.

### 3.2 Gradient Computation

Compute partial derivatives via smoothed Sobel filters:

```python
I_x = I_smooth * Sobel_x
I_y = I_smooth * Sobel_y
G = sqrt(I_x**2 + I_y**2)
θ = arctan2(I_y, I_x)
```

### 3.3 Non-Maximum Suppression

For each pixel, compare gradient magnitude to its two neighbors along θ direction. If not a local maximum, zero it out.

### 3.4 Double Threshold + Hysteresis

- Strong edges: G ≥ T_high
- Weak edges: T_low ≤ G < T_high
- Discard G < T_low
- Link weak edges to strong edges if connected (8-connectivity).

### 4. Code & Practical Application

### 4.1 Canny from Scratch (NumPy)

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma):
    ax = np.arange(-size//2 + 1, size//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2)/(2*sigma**2))
    return kernel / np.sum(kernel)

def convolve2d(img, kernel):
    pad = kernel.shape[0]//2
    img_p = np.pad(img, pad, mode='reflect')
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i,j] = np.sum(img_p[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    return out

# 1. Load & grayscale
I = np.array(Image.open('cat.jpg').convert('L'), dtype=np.float32)

# 2. Smooth
gk = gaussian_kernel(5, sigma=1.4)
I_s = convolve2d(I, gk)

# 3. Gradients
sobel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=np.float32)
sobel_y = sobel_x.T * -1
I_x = convolve2d(I_s, sobel_x)
I_y = convolve2d(I_s, sobel_y)
G   = np.hypot(I_x, I_y)
θ   = np.arctan2(I_y, I_x)

# 4. Non-max suppression (simplified)
M, N = I.shape
nms = np.zeros_like(G)
angle = θ * (180.0/np.pi)
angle[angle<0] += 180

for i in range(1, M-1):
    for j in range(1, N-1):
        q = 255; r = 255
        # determine neighbors based on angle
        if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
            q = G[i, j+1]; r = G[i, j-1]
        elif (22.5 <= angle[i,j] < 67.5):
            q = G[i+1, j-1]; r = G[i-1, j+1]
        elif (67.5 <= angle[i,j] < 112.5):
            q = G[i+1, j]; r = G[i-1, j]
        else:
            q = G[i-1, j-1]; r = G[i+1, j+1]
        if G[i,j] >= q and G[i,j] >= r:
            nms[i,j] = G[i,j]

# 5. Hysteresis
high, low = 0.2*nms.max(), 0.1*nms.max()
res = np.zeros_like(nms, dtype=np.uint8)
strong = (nms >= high)
weak   = (nms >= low) & (nms < high)
res[strong] = 255

# link weak to strong
for i in range(1, M-1):
    for j in range(1, N-1):
        if weak[i,j] and np.any(strong[i-1:i+2, j-1:j+2]):
            res[i,j] = 255

plt.imshow(res, cmap='gray')
plt.title('Canny Edges'); plt.axis('off')
plt.show()
```

### 4.2 OpenCV Built-in

```python
import cv2

I = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(I, threshold1=50, threshold2=150, apertureSize=3)
cv2.imshow('Canny', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 5. Visualization / Geometry

- **Gaussian smoothing**: suppresses high-frequency noise (“blurs” small speckles).
- **Gradient orientation**: represents edge direction.
- **Non-max suppression**: thins edges by selecting only pixels that are local maxima along their gradient direction.
- **Hysteresis**: connects weak segments to strong edges, yielding continuous boundaries.

```
Raw Image → Gaussian Blur → Gradient Magnitude & Direction
    → Non-Max Suppression → Double Threshold → Edge Map
```

### 6. Common Pitfalls & Tips

- Picking σ too small leaves noise; too large oversmooths edges.
- Thresholds T_low/T_high require grid search per dataset.
- Skipping non-max suppression yields thick, fuzzy edges.
- Forgetting 8-connectivity in hysteresis breaks contours.
- Using single threshold misses low-contrast edges.

### 7. Interview-Ready Insights

- Explain why Canny is “optimal” per criteria: low error rate, good localization, single response.
- Compare Canny with Marr–Hildreth (Laplacian of Gaussian) and Difference of Gaussians:
    - Marr–Hildreth uses zero-crossings of ∇²G; DoG approximates it efficiently.
- Discuss computational cost vs. quality trade-offs.
- Show how early CNN filters resemble first-derivative kernels; deeper layers build on these primitives.

### 8. Practice Exercises

1. **Laplacian of Gaussian (LoG)**
    - Create a 7×7 LoG kernel with σ=1. Use it to detect edges via zero-crossings.
    - Visualize the kernel and the edge map.
2. **DoG vs. LoG**
    - Implement Difference of Gaussians: blur with σ=1 and σ=2, subtract.
    - Compare edge maps from DoG and LoG.
3. **Adaptive Canny**
    - Replace fixed thresholds with mean±k·std of gradient magnitudes.
    - Test on images with varying contrast; observe improvements.
4. **Visualize Learned CNN Filters**
    - Load a pretrained PyTorch model (e.g., VGG16).
    - Extract first-layer filters and plot them; confirm they look like edge detectors.
    - Code hint:
        
        ```python
        import torchvision.models as models
        import matplotlib.pyplot as plt
        
        vgg = models.vgg16(pretrained=True)
        filters = vgg.features[0].weight.data.clone()
        # plot first 16 filters
        ```
        
5. **Edge-guided Image Segmentation**
    - Use your Canny map as a mask to guide a simple K-means segmentation on color pixels:
        - Strong edges define cluster boundaries—prevent merging across them.

---

## Padding in Convolution Operations

### 1. Direct Definition

Padding inserts additional pixels around the border of an image or feature map before applying a convolutional filter. It controls how border information is handled and determines the spatial dimensions of the output.

### 2. Intuition Behind Padding

Convolving a filter without padding shrinks the output, because border pixels are covered fewer times. Padding preserves edge information by giving every input pixel an equal chance to contribute to the result.

### 3. Types of Padding

| Padding type | Description | Output size relative to input |
| --- | --- | --- |
| valid | No padding; only fully overlapping positions | smaller |
| same | Pads so output size equals input size | same |
| full | Pads so every filter position covers some input | larger |
| reflect | Mirrors border pixels | varies |
| replicate | Repeats edge pixel values | varies |

### 4. Output Dimension Formula

For a 1D signal of length (I), filter size (K), padding (P), and stride (S), the output length (O) is:

$[ O = \frac{I - K + 2P}{S} + 1 ]$

For 2D images apply this formula independently to height and width.

### 5. Code Examples

### 5.1 NumPy Convolution with Zero Padding

```python
import numpy as np

def pad_image(img, pad):
    return np.pad(img, ((pad, pad), (pad, pad)), mode='constant')

def conv2d(img, kernel, padding=0, stride=1):
    img_p = pad_image(img, padding)
    k_h, k_w = kernel.shape
    out_h = (img_p.shape[0] - k_h)//stride + 1
    out_w = (img_p.shape[1] - k_w)//stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(0, out_h):
        for j in range(0, out_w):
            region = img_p[i*stride:i*stride+k_h, j*stride:j*stride+k_w]
            out[i, j] = np.sum(region * kernel)
    return out
```

### 5.2 PyTorch Convolution Layer

```python
import torch
import torch.nn as nn

conv_same = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
conv_valid = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
```

### 6. Common Pitfalls & Tips

- Using zero padding on images with dark borders can create artificial edges.
- Overpadding increases computation and may dilute feature locality.
- Choosing same padding by default can hide how kernel size affects receptive field.
- Forgetting to adjust stride when padding changes output dimensions.

### 7. Interview-Ready Insights

- Explain why same padding with odd kernel sizes centers the filter symmetrically.
- Discuss how reflect and replicate padding reduce border artifacts compared to zero padding.
- Compare padding strategies in spatial convolutions versus dilated and transposed convolutions.
- Illustrate impact of padding on receptive field growth in deep networks like ResNet.

### 8. Practice Exercises

1. Implement a 5×5 convolution with replicate padding and plot results on a sample image.
2. Compare edge maps from valid versus same padding when applying a Sobel filter.
3. Modify a simple CNN to use reflect padding and evaluate its effect on classification accuracy.
4. Compute effective receptive field size after three stacked 3×3 convolutions with same padding.

---

## Strided Convolution

### 1. Direct Definition

Strided convolution applies a convolutional filter over an input by moving the filter window by a fixed step size (stride) greater than one in the spatial dimensions. This down-samples the feature map, reducing its width and height proportionally to the stride.

### 2. Intuition Behind Stride

When you slide a filter one pixel at a time (stride = 1), you build a dense response map. By increasing the stride, you skip positions, effectively “zooming out” and summarizing neighborhoods. Strided convolution thus merges feature extraction and spatial downsampling into one operation.

### 3. Output Dimension Formula

For an input of size (I), kernel size (K), padding (P), and stride (S), the output size (O) is:

$[ O = \left\lfloor \frac{I - K + 2P}{S} \right\rfloor + 1 ]$

Apply this formula separately to height and width in 2D.

### 4. Code Examples

### 4.1 NumPy Convolution with Stride

```python
import numpy as np

def conv2d_strided(img, kernel, padding=0, stride=1):
    # Pad image
    img_p = np.pad(img, ((padding, padding), (padding, padding)), mode='constant')
    k_h, k_w = kernel.shape
    out_h = (img_p.shape[0] - k_h)//stride + 1
    out_w = (img_p.shape[1] - k_w)//stride + 1
    out = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            y = i * stride
            x = j * stride
            region = img_p[y:y+k_h, x:x+k_w]
            out[i, j] = np.sum(region * kernel)

    return out

# Example usage
image  = np.random.rand(32, 32)
kernel = np.ones((3, 3)) / 9
feature_map = conv2d_strided(image, kernel, padding=1, stride=2)
print(feature_map.shape)  # (16, 16)
```

### 4.2 PyTorch Convolution Layer

```python
import torch
import torch.nn as nn

# Convolution with stride 2 down-samples by 2
conv_strided = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)

# Input: batch of 8 RGB images 64×64
x = torch.randn(8, 3, 64, 64)
out = conv_strided(x)
print(out.shape)  # torch.Size([8, 16, 32, 32])
```

### 5. Common Pitfalls & Tips

- Large strides can discard fine details and introduce aliasing artifacts.
- Always adjust padding when changing stride to avoid unintended dimension shifts.
- Mixing strides and pooling layers may over-downsample; track the receptive field carefully.
- When visualizing feature maps, remember that spatial correspondence is coarser with higher stride.

### 6. Interview-Ready Insights

- Contrast strided convolution with pooling: convolution learns weights while pooling uses fixed operations.
- Explain how stride impacts the receptive field growth across layers.
- Discuss the dual role of stride in feature abstraction and dimension reduction in modern architectures (e.g., ResNet, MobileNet).
- Outline how transposed (deconvolutional) layers invert strided convolutions to up-sample.

### 7. Practice Exercises

1. Implement a small CNN using only strided convolutions (no pooling) to down-sample by factor 8 and verify output dimensions.
2. Compare feature maps from stride = 2 and pooling + stride = 1; visualize the difference in learned patterns.
3. Compute the effective receptive field after stacking four 3×3 convolutions with stride = 2 each.
4. Replace strided convolutions with dilated convolutions in a toy network and observe how feature resolution and receptive field change.

---

## Convolutions over Volumes

### 1. Direct Definition

A volumetric convolution (often called 3D convolution) extends the 2D convolution operation into three spatial dimensions—height, width, and depth (which can represent depth slices in a volume or temporal frames in a video). Instead of sliding a 2D kernel over a matrix, you slide a 3D kernel over a cuboid of data, producing feature maps that capture relationships across all three axes.

### 2. Intuition Behind 3D Convolution

When you apply a 2D filter to an image, you learn spatial patterns like edges or textures. Extending that filter through a third dimension lets you capture:

- Continuity between adjacent slices in a medical scan
- Motion and appearance changes across video frames
- Correlations across spectral bands in hyperspectral imagery

A 3D kernel “sees” a small block of consecutive slices and learns features that span depth and spatial dimensions simultaneously.

### 3. Output Dimension Formula

For an input volume of size $(D \times H \times W)$, kernel size $(K_d \times K_h \times K_w)$, padding $(P_d, P_h, P_w)$, and stride $(S_d, S_h, S_w)$, the output volume dimensions $((O_d, O_h, O_w))$ are:

$[ O_d = \left\lfloor \frac{D - K_d + 2P_d}{S_d} \right\rfloor + 1 ]$ $[ O_h = \left\lfloor \frac{H - K_h + 2P_h}{S_h} \right\rfloor + 1 ]$ $[ O_w = \left\lfloor \frac{W - K_w + 2P_w}{S_w} \right\rfloor + 1 ]$

### 4. Code Examples

### 4.1 NumPy: Naïve 3D Convolution

```python
import numpy as np

def conv3d(volume, kernel, padding=(0,0,0), stride=(1,1,1)):
    D, H, W = volume.shape
    Kd, Kh, Kw = kernel.shape
    Pd, Ph, Pw = padding
    Sd, Sh, Sw = stride

    # pad volume
    vol_p = np.pad(volume,
                   ((Pd, Pd), (Ph, Ph), (Pw, Pw)),
                   mode='constant')

    Od = (D - Kd + 2*Pd)//Sd + 1
    Oh = (H - Kh + 2*Ph)//Sh + 1
    Ow = (W - Kw + 2*Pw)//Sw + 1

    out = np.zeros((Od, Oh, Ow))

    for z in range(Od):
        for y in range(Oh):
            for x in range(Ow):
                z0 = z * Sd; y0 = y * Sh; x0 = x * Sw
                region = vol_p[z0:z0+Kd, y0:y0+Kh, x0:x0+Kw]
                out[z, y, x] = np.sum(region * kernel)
    return out

# Example: random 8×32×32 volume and 3×5×5 kernel
volume = np.random.rand(8, 32, 32)
kernel = np.ones((3, 5, 5)) / 75
feature_map = conv3d(volume, kernel, padding=(1,2,2), stride=(2,1,1))
print(feature_map.shape)  # (4, 32, 32)
```

### 4.2 PyTorch: Built-in 3D Convolution

```python
import torch
import torch.nn as nn

# Define a 3D conv: in_channels=1, out_channels=8, kernel=3×3×3, stride=2 in depth
conv3d = nn.Conv3d(
    in_channels=1,
    out_channels=8,
    kernel_size=(3, 3, 3),
    stride=(2, 1, 1),
    padding=(1, 1, 1)
)

# Input: batch of 4 volumes, each 16×64×64
x = torch.randn(4, 1, 16, 64, 64)
out = conv3d(x)
print(out.shape)  # torch.Size([4, 8, 8, 64, 64])
```

### 5. Common Pitfalls & Tips

- Volumetric convs multiply computational and memory cost by depth—use judiciously or on small volumes.
- Large 3D kernels capture broader context but risk overfitting if data is scarce.
- Factorizing a 3D conv into spatial (2D) + temporal (1D) convolutions (the “(2+1)D” trick) often reduces parameters and improves generalization.
- Keep an eye on padding: mismatched padding per axis can distort your volume’s geometry.

### 6. Interview-Ready Insights

- Explain the trade-off between 3D conv and sequential 2D conv + RNN on videos:
    - 3D convs learn unified spatiotemporal filters, while 2D convs + RNNs separate spatial and temporal processing.
- Discuss applications:
    - Medical imaging (CT/MRI segmentation with U-Net3D)
    - Action recognition in videos (e.g., C3D, I3D architectures)
- Compare full 3D conv with pseudo-3D (P3D) and factorized 2D + 1D convs in terms of receptive field and parameter efficiency.

### 7. Practice Exercises

1. Build a small 3D CNN in PyTorch that downsamples a 16-slice volume to a single feature map and verify dimensions.
2. Implement the (2+1)D factorization: first a 1×3×3 spatial conv, then a 3×1×1 temporal conv; compare parameter count with full 3×3×3.
3. Visualize learned 3D kernels by plotting central slices of each filter in your trained network.
4. Apply a 3D U-Net on a toy MRI dataset: experiment with different depth levels and report segmentation accuracy.

---

## One Convolutional Layer

### 1. Direct Definition

A single convolutional layer consists of a set of learnable 2D filters (kernels) that slide over an input volume, computing dot-products to produce feature maps, then adds a bias per map and applies a pointwise activation function.

### 2. Intuition Behind One Layer

Every filter acts like a learned template that “glides” across spatial dimensions, detecting patterns (edges, textures, motifs) in all input channels simultaneously. Weight sharing lets the network spot the same feature at different locations, while local connectivity ensures each output unit only “sees” a small receptive field of the input.

### 3. Shape & Output Dimensions

- Input volume shape: $(C_{\text{in}}\times H\times W)$
- Number of filters: $(C_{\text{out}})$
- Kernel size: $(K_h\times K_w)$
- Stride: (S)
- Padding: (P)

Output feature map dimensions $(H_{\text{out}})$ and $(W_{\text{out}})$:

$[ H_{\text{out}}$ = $\left\lfloor \frac{H + 2P - K_h}{S} \right\rfloor$ + 1 $\quad W_{\text{out}} = \left\lfloor \frac{W + 2P - K_w}{S} \right\rfloor + 1 ]$

Output volume shape: $(C_{\text{out}}\times H_{\text{out}}\times W_{\text{out}})$.

### 4. Mathematical Formulation

For input $(X[c,i,j])$, weights $(W[o,c,u,v])$, $bias (b[o])$, stride (S), and padding (P):

$[ Y[o,i,j]$ = $\sum_{c=0}^{C_{\text{in}}-1}\sum_{u=0}^{K_h-1}\sum_{v=0}^{K_w-1} W[o,c,u,v] X\bigl[c, i\cdot S + u - P, j\cdot S + v - P\bigr] + b[o] ]$

Then apply activation (A) elementwise:

$[ Z[o,i,j] = A\bigl(Y[o,i,j]\bigr) ]$

### 5. Code Examples

### 5.1 NumPy Implementation (Single Layer Forward)

```python
import numpy as np

def conv_layer_forward(X, W, b, stride=1, pad=0):
    C_out, C_in, Kh, Kw = W.shape
    C_in2, H, Wd = X.shape
    assert C_in2 == C_in

    H_out = (H + 2*pad - Kh)//stride + 1
    W_out = (Wd + 2*pad - Kw)//stride + 1
    # pad input
    X_p = np.pad(X, ((0,0),(pad,pad),(pad,pad)), mode='constant')
    Y = np.zeros((C_out, H_out, W_out))

    for o in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                hs = i*stride; ws = j*stride
                patch = X_p[:, hs:hs+Kh, ws:ws+Kw]
                Y[o,i,j] = np.sum(W[o]*patch) + b[o]
    # apply ReLU activation
    Z = np.maximum(0, Y)
    return Z

# example shapes
X = np.random.randn(3, 32, 32)           # 3-channel 32×32
W = np.random.randn(8, 3, 5, 5) * 0.1    # 8 filters of 3×5×5
b = np.zeros(8)
out = conv_layer_forward(X, W, b, stride=1, pad=2)
print(out.shape)  # (8, 32, 32)
```

### 5.2 PyTorch Module

```python
import torch
import torch.nn as nn

# one conv layer: in 3 channels, out 8 channels, 5×5 kernel, pad=2, stride=1
conv = nn.Conv2d(in_channels=3,
                 out_channels=8,
                 kernel_size=(5,5),
                 stride=1,
                 padding=2)
# forward pass on batch of 4 images 3×32×32
x = torch.randn(4, 3, 32, 32)
z = conv(x)           # has bias and no activation
z_act = torch.relu(z) # apply ReLU separately
print(z_act.shape)    # torch.Size([4, 8, 32, 32])
```

### 6. Visualization / Geometry

A sliding window of size $(K_h\times K_w)$ moves in steps of (S). At each position it computes a weighted sum across all (C_{\text{in}}) channels, producing one scalar in each of the $(C_{\text{out}})$ feature maps. Zero-padding of width (P) ensures the filter covers border pixels without shrinking the map.

```
Input Volume ──▶ [Pad] ──▶ [Slide W1] ──▶ Weighted Sum ──▶ +b1 ──▶ A  ──▶ Feature Map 1
              └─▶ [Slide W2] ──▶ …                     +b2           Feature Map 2
              └─▶ [Slide Wc]
```

### 7. Interview-Ready Insights

- Weight sharing drastically reduces parameters compared to a fully connected layer over the same receptive field.
- Convolution is translation-equivariant: shifting the input shifts the output accordingly.
- Activation function choice (ReLU, LeakyReLU, ELU) affects gradient flow and sparsity of feature maps.
- Batch normalization or bias folding can be applied after convolution to stabilize training.

### 8. Practice Exercises

1. Replace ReLU with a custom activation (e.g., LeakyReLU) in your NumPy forward and observe feature map differences.
2. Modify the NumPy implementation to return intermediate patches for visual inspection of learned filter matches.
3. Stack two conv layers (first 3→8, then 8→16 channels) and compute the overall receptive field size at the output.
4. Implement and time the backward pass for the single layer to compute gradients w.r.t. (W), (b), and (X).

---

## Simple Convolution Layer Example

### 1. Direct Definition

A simple convolutional layer applies a set of learnable filters across an input tensor, computing weighted sums over local patches, adding a bias, and then applying an activation function to produce output feature maps.

### 2. Intuition

Each filter acts like a small window that scans over the input volume to detect a specific pattern. At every position it multiplies the overlapping patch by its weights, sums the result, adds its bias, and then “fires” through an activation like ReLU if the response is positive.

### 3. Shape & Output Dimensions

- Input shape : Cin × H × W
- Number of filters : Cout
- Kernel size : Kh × Kw
- Stride : S
- Padding : P

Output height H_out = floor((H + 2×P − Kh) / S) + 1

Output width W_out = floor((W + 2×P − Kw) / S) + 1

Output shape : Cout × H_out × W_out

### 4. Mathematical Formulation

For each output channel o and spatial location (i, j):

Y[o, i, j] =

sum over c=0…Cin–1 of sum over u=0…Kh–1 of sum over v=0…Kw–1 of

W[o, c, u, v] × X[c, i×S + u − P, j×S + v − P] 

+ b[o]

Activated output Z[o, i, j] = max(0, Y[o, i, j]) # ReLU

### 5. Code Examples

### 5.1 NumPy Implementation

```python
import numpy as np

def simple_conv2d(input, weights, bias, stride=1, pad=0):
    Cin, H, W = input.shape
    Cout, _, Kh, Kw = weights.shape
    H_out = (H + 2*pad - Kh) // stride + 1
    W_out = (W + 2*pad - Kw) // stride + 1

    x_p = np.pad(input, ((0,0),(pad,pad),(pad,pad)), mode='constant')
    y = np.zeros((Cout, H_out, W_out))

    for o in range(Cout):
        for i in range(H_out):
            for j in range(W_out):
                h0 = i * stride
                w0 = j * stride
                patch = x_p[:, h0:h0+Kh, w0:w0+Kw]
                y[o, i, j] = np.sum(patch * weights[o]) + bias[o]

    # ReLU activation
    z = np.maximum(0, y)
    return z

# Example
input  = np.random.randn(1, 28, 28)          # single-channel 28×28
weights = np.random.randn(4, 1, 3, 3) * 0.1  # 4 filters
bias    = np.zeros(4)
output  = simple_conv2d(input, weights, bias, stride=1, pad=1)
print(output.shape)  # (4, 28, 28)
```

### 5.2 PyTorch Module

```python
import torch
import torch.nn as nn

conv = nn.Conv2d(
    in_channels=1,
    out_channels=4,
    kernel_size=3,
    stride=1,
    padding=1
)

x      = torch.randn(1, 1, 28, 28)  # batch of 1
z      = conv(x)                    # convolution + bias
z_act  = torch.relu(z)              # apply ReLU
print(z_act.shape)                  # torch.Size([1, 4, 28, 28])
```

### 6. Visualization / Geometry

With padding = 1 and stride = 1, a 3×3 filter starts at the padded top-left corner, slides across each row, and ends at the padded bottom-right. Each output map highlights where its filter pattern appears in the input.

### 7. Interview-Ready Insights

- Weight sharing means total parameters = Cout × Cin × Kh × Kw + Cout, much less than a fully connected layer.
- Zero padding preserves border information and controls output size.
- Increasing stride both down-samples spatial dimensions and reduces computation.
- Applying ReLU introduces sparsity, speeds up training, and mitigates vanishing gradients.

### 8. Practice Exercises

- Replace ReLU with sigmoid and compare activation distributions.
- Set stride to 2, observe how output dimensions shrink by half.
- Stack two such layers (1→4, then 4→8 channels) and compute the overall receptive field.
- Visualize each 3×3 filter as a grayscale patch to see what patterns the layer learns.

---

## Pooling Layers

### 1. Direct Definition

Pooling layers reduce the spatial dimensions of feature maps by summarizing local neighborhoods using a fixed operation—typically max or average—while preserving the depth (number of channels).

### 2. Intuition Behind Pooling

By condensing each patch of activations into a single value, pooling layers introduce translation invariance and help the network focus on the most salient features. This step also cuts down computation and limits overfitting by reducing the number of activations processed in deeper layers.

### 3. Common Pooling Types

| Pooling Type | Operation | Output size relative to input |
| --- | --- | --- |
| max | selects maximum value in patch | smaller |
| average | computes average value in patch | smaller |
| global max | selects maximum over entire map | 1×1 per channel |
| global average | computes average over entire map | 1×1 per channel |

### 4. Output Dimension

For a 2D input of height H and width W, pool size Kh×Kw, stride S, and no padding, the output height H_out and width W_out are:

H_out = floor((H − Kh) / S) + 1

W_out = floor((W − Kw) / S) + 1

Using padding or ceil_mode changes these calculations accordingly.

### 5. Code Examples

### 5.1 NumPy Implementation of Max Pooling

```python
import numpy as np

def max_pool2d(x, pool_size=2, stride=2):
    C, H, W = x.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    out = np.zeros((C, H_out, W_out))
    for c in range(C):
        for i in range(H_out):
            for j in range(W_out):
                h0 = i * stride
                w0 = j * stride
                patch = x[c, h0:h0+pool_size, w0:w0+pool_size]
                out[c, i, j] = np.max(patch)
    return out

# Example usage
x = np.random.randn(3, 32, 32)
pooled = max_pool2d(x, pool_size=2, stride=2)
print(pooled.shape)  # (3, 16, 16)
```

### 5.2 PyTorch Module

```python
import torch
import torch.nn as nn

x = torch.randn(4, 3, 32, 32)  # batch of 4 feature maps
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

y_max = max_pool(x)
y_avg = avg_pool(x)
print(y_max.shape)  # torch.Size([4, 3, 16, 16])
print(y_avg.shape)  # torch.Size([4, 3, 16, 16])
```

### 6. Common Pitfalls & Tips

- Overusing pooling can discard fine-grained spatial information and harm tasks that require precise localization.
- Zero padding before pooling may introduce artificial maxima at the borders.
- Global pooling collapses all spatial structure, so it’s best used just before a classification or regression head.
- Choosing stride smaller than the pool size creates overlapping regions and highly correlated outputs.

### 7. Interview-Ready Insights

- Pooling vs. strided convolution: pooling uses fixed functions, while strided convolution learns weights, offering greater modeling flexibility.
- Translation invariance emerges because small shifts within a pooling window yield the same output.
- Global average pooling drastically reduces parameters and enforces a direct mapping between feature maps and classes, as seen in architectures like ResNet.
- Interpreting max pooling as a form of soft “attention” helps explain why it excels at capturing strong activations.

### 8. Practice Exercises

- Implement average pooling from scratch in NumPy and compare its outputs against max pooling.
- Experiment with overlapping pooling by setting stride < pool size and visualize activation patterns.
- Replace pooling layers in a simple CNN with strided convolutions and measure the impact on accuracy and parameter count.
- Apply global average pooling to the final feature map of a pretrained model and connect it directly to a linear classifier.

---

## Simple CNN Example

### 1. Overview

We’ll build and train a lightweight convolutional neural network on the MNIST digit-classification task. This model has two convolutional blocks (conv → ReLU → pooling) followed by a fully connected layer. It illustrates the core building blocks of any image-based CNN.

### 2. Architecture

- Input: 1×28×28 grayscale image
- ConvBlock1
    - Conv2d: in=1, out=16, kernel=3×3, padding=1 → ReLU
    - MaxPool2d: kernel=2×2, stride=2
- ConvBlock2
    - Conv2d: in=16, out=32, kernel=3×3, padding=1 → ReLU
    - MaxPool2d: kernel=2×2, stride=2
- Fully Connected
    - Flatten → Linear(32×7×7 → 10)

Output is a 10-dim vector of class scores.

### 3. PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # ConvBlock1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
                               kernel_size=3, stride=1, padding=1)
        # ConvBlock2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        # Fully connected layer
        self.fc = nn.Linear(in_features=32 * 7 * 7, out_features=10)

    def forward(self, x):
        # Block 1: conv → ReLU → pool
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # Block 2: conv → ReLU → pool
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # Flatten and fully connect
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Instantiate and check shapes
model = SimpleCNN()
dummy_input = torch.randn(4, 1, 28, 28)  # batch of 4 MNIST images
output = model(dummy_input)
print(output.shape)  # torch.Size([4, 10])
```

### 4. Training Loop (Skeleton)

```python
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.13,), (0.31,))
])
train_ds = datasets.MNIST(root='mnist_data', train=True,
                          download=True, transform=transform)
train_ld = DataLoader(train_ds, batch_size=64, shuffle=True)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Train for one epoch
model.train()
for images, labels in train_ld:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
print(f'Finished 1 epoch, last batch loss: {loss.item():.4f}')
```

### 5. Explanation of Key Steps

- **Padding=1** in each conv layer preserves the 28×28 spatial size before pooling.
- **MaxPool2d(2×2)** halves H and W after each block: 28→14→7.
- **ReLU** adds nonlinearity and sparsity.
- **Flatten** reshapes the 32×7×7 tensor into a 1D vector of size 1568 for the linear layer.
- **CrossEntropyLoss** combines softmax + negative log-likelihood for multi-class classification.

### 6. Interview-Ready Insights

- Weight sharing in conv layers reduces parameters: here total conv params =(1×3×3×16) + (16×3×3×32) + biases.
- Pooling introduces translation invariance and reduces overfitting.
- You can replace pooling with strided conv to learn down-sampling.
- Batch normalization after each conv can speed up convergence and improve stability.

### 7. Practice Exercises

1. Add a third conv block (32→64 filters) and measure test accuracy improvements.
2. Swap max pooling for average pooling and compare performance.
3. Insert a Dropout layer before the fully connected layer to mitigate overfitting.
4. Train the same architecture on Fashion-MNIST and observe differences in convergence.

---

## Why Convolutions?

### 1. Direct Definition

Convolutions replace fully connected layers for processing images (and other spatial data) by sliding small, learnable filters over the input. Each filter performs a weighted sum over a local patch, producing a feature map. This operation exploits three key principles—sparse connectivity, parameter sharing, and translation equivariance—to efficiently learn spatial hierarchies of features.

### 2. Concept Intuition

- **Local patterns matter**
    
    Images are not random grids—nearby pixels form edges, textures, corners. Convolutions focus each filter on a small “receptive field” (e.g., 3×3), detecting local motifs that recur across the entire image.
    
- **Parameter sharing**
    
    A single filter with one set of weights scans every location. Instead of learning separate weights for each pixel pair (as in a fully connected layer), the network reuses the same filter thousands of times, cutting parameter count dramatically.
    
- **Translation equivariance**
    
    If an edge appears at the top-left or bottom-right, the same filter will respond. Convolutions inherently respect spatial shifts, so the model generalizes to patterns anywhere in the image.
    
- **Hierarchical feature building**
    
    Stacking conv layers grows the effective receptive field. Early layers detect simple edges, mid-layers combine edges into textures or shapes, deeper layers assemble object parts into semantic concepts.
    

### 3. Mathematical Breakdown

Let

- X[c, i, j] be the input at channel c, row i, column j
- W[o, c, u, v] be the weight of filter o at channel c, offset (u,v)
- b[o] be the bias for filter o
- S be the stride, P the zero-padding

Then a single output activation Y[o, i, j] is computed as:

```python
Y[o, i, j] = 0
for c in range(Cin):
    for u in range(Kh):
        for v in range(Kw):
            # map output (i,j) back to input coordinate
            in_i = i * S + u - P
            in_j = j * S + v - P
            if 0 <= in_i < H and 0 <= in_j < W:
                Y[o, i, j] += W[o, c, u, v] * X[c, in_i, in_j]
Y[o, i, j] += b[o]
Z[o, i, j] = activation(Y[o, i, j])  # e.g., ReLU: max(0, Y)
```

Output spatial dimensions H_out, W_out:

```python
H_out = floor((H + 2*P - Kh) / S) + 1
W_out = floor((W + 2*P - Kw) / S) + 1
```

### 4. Code & Practical Application

### 4.1 Comparing Conv vs. Fully Connected on a Tiny “Image”

```python
import numpy as np

# define a 4×4 “image” with 1 channel
X = np.arange(16).reshape(1,4,4).astype(float)

# fully connected: flatten → weight matrix 16→8 outputs
W_fc = np.random.randn(8, 16)
b_fc = np.random.randn(8)
fc_out = W_fc.dot(X.flatten()) + b_fc

# convolution: use two 2×2 filters, stride=2, no padding
W_conv = np.random.randn(2, 1, 2, 2)  # out_channels=2, in=1, 2×2 kernels
b_conv = np.random.randn(2)
conv_out = np.zeros((2,2,2))          # compute output shape manually

# perform conv
for o in range(2):
    for i in range(2):
        for j in range(2):
            patch = X[:, i*2:i*2+2, j*2:j*2+2]
            conv_out[o,i,j] = np.sum(W_conv[o] * patch) + b_conv[o]

print("FC out shape:", fc_out.shape)      # (8,)
print("Conv out shape:", conv_out.shape)  # (2,2,2)
```

- FC uses **16×8 = 128** weights.
- Conv uses **2×1×2×2 = 8** weights.

Convolution is drastically more parameter-efficient.

### 4.2 Real-World Workflow

1. **Data loading**: batch of images → shape (B, C, H, W).
2. **Normalization & augmentation**.
3. **Forward pass**: multiple conv + activation + pooling blocks.
4. **Feature extraction**: final conv block outputs high-level feature maps.
5. **Classification head**: global pooling + linear layer → logits.

### 5. Visualization / Geometry

```
 Input Image (H×W)
        ↓ 3×3 filter slides
 Feature Maps (H_out×W_out)
        ↓ stack many filters
 Multi-channel tensor
```

- A filter’s **receptive field** on the input grows with depth.
- Visualizing the learned weights of a 3×3 filter often reveals edge-like or Gabor-like patterns.

Plotting a filter’s weights:

```python
import matplotlib.pyplot as plt

# assume W_conv[0,0] is a 3×3 kernel
plt.imshow(W_conv[0,0], cmap='gray')
plt.title('Filter 0 Channel 0')
plt.colorbar()
plt.show()
```

### 6. Common Pitfalls & Tips

- **No padding** shrinks outputs and loses border info.
- **Large strides** skip detail and cause aliasing.
- **Too many filters** early on bloats memory with low-level features.
- **Incorrect input ordering** (H×W×C vs. C×H×W) causes shape mismatches.
- **Not tracking receptive field** leads to unexpected coverage—use simple scripts to calculate it.

### 7. Interview-Ready Insights

- **Sparse interactions**: each output depends on a small local region, not the entire input.
- **Parameter sharing**: a filter is reused across all spatial locations, giving translation equivariance.
- **Equivariance vs. invariance**: conv layers shift responses consistently when inputs shift; pooling + deeper layers build invariance to small shifts.
- **Receptive field growth**: stacking L layers of 3×3 convs (stride=1, pad=1) gives effective receptive field of size 2L+1.
- **Depthwise separable convolutions** reduce cost by splitting spatial vs. channel mixing (used in MobileNet).

### 8. Practice Exercises

1. **Parameter comparison**
    - For a 64×64×3 input, compute parameters for:
        - a FC layer to 100 outputs
        - a conv layer with 16 filters of size 5×5
    - Verify conv has far fewer weights.
2. **Receptive field script**
    - Write a function that takes a sequence of (kernel, stride, padding) and returns the final receptive field size and output dimensions.
3. **Visualizing learned filters**
    - Train a simple CNN on CIFAR-10 for one epoch.
    - Extract the first conv layer’s filters and plot them in a grid.
4. **Conv vs. strided conv vs. pooling**
    - Build three small models that down-sample by factor 2 using each method.
    - Compare final feature map shapes and number of learnable parameters.
5. **Translation test**
    - Take an image with a single centered object.
    - Shift it by a few pixels and pass both through a trained conv filter.
    - Plot filter responses to confirm translation equivariance.

---