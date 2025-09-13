# DL_c4_m3

## Object Localization

### 1. Direct definition

Object localization is the task of predicting the position of a single object within an image by outputting a bounding‐box—typically represented as (x₁, y₁, x₂, y₂) or (center_x, center_y, width, height)—alongside its class label.

### 2. Concept intuition

At its core, localization asks: “Where is the object?”

- Instead of merely answering “What is it?” (classification), we regress real‐valued coordinates.
- This bridges perception with action: robotics needs to know where to grasp, autonomous cars must know where pedestrians are.
- We transform spatial features from convolutional layers into geometric predictions.

Why it matters:

- Serves as the foundation for full object detection systems (e.g., YOLO, Faster R-CNN).
- Drives key metrics like Intersection over Union (IoU) and mean Average Precision (mAP), which measure both classification and localization quality.

Prerequisites & refreshers

- Linear regression (minimize squared error between prediction and target).
- Understanding image coordinates: origin at top-left, x→right, y→down.

### 3. Mathematical breakdown

We treat localization as a regression problem. For one example:

```
pred = [x1_pred, y1_pred, x2_pred, y2_pred]
true = [x1_true, y1_true, x2_true, y2_true]
```

A common loss is Mean Squared Error (MSE):

```python
# one-sample localization loss
loss_loc = ((x1_pred - x1_true)**2
          + (y1_pred - y1_true)**2
          + (x2_pred - x2_true)**2
          + (y2_pred - y2_true)**2) / 4.0
```

Breakdown of variables

- x1, y1: top-left corner coordinates
- x2, y2: bottom-right corner coordinates
- All coordinates normalized to [0, 1] by dividing by image width/height

Why it works

- Penalizes Euclidean distance between predicted and ground-truth corners.
- Averaging (division by 4) keeps loss magnitude on a comparable scale to classification losses.

Smooth L1 loss (Huber) is often preferred to reduce sensitivity to outliers:

```python
def smooth_L1_diff(a, b, beta=1.0):
    diff = abs(a - b)
    if diff < beta:
        return 0.5 * diff**2 / beta
    else:
        return diff - 0.5 * beta

loss_loc_smooth = sum(smooth_L1_diff(p, t) for p, t in zip(pred, true)) / 4.0
```

### 4. Code & practical application

Below is a minimal PyTorch example: a CNN predicting four box coordinates on synthetic images.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2

# 1. Synthetic dataset: single white square on black background
class SquareDataset(Dataset):
    def __init__(self, n_samples=500, img_size=64, sq_size=16):
        self.n, self.S, self.sq = n_samples, img_size, sq_size
    def __len__(self): return self.n
    def __getitem__(self, idx):
        img = np.zeros((self.S, self.S), dtype=np.uint8)
        x = np.random.randint(0, self.S - self.sq)
        y = np.random.randint(0, self.S - self.sq)
        img[y:y+self.sq, x:x+self.sq] = 255
        # normalize and add channel
        img = img.astype(np.float32) / 255.0
        img = img[None,:,:]
        # normalize coords to [0,1]
        box = np.array([x, y, x+self.sq, y+self.sq], dtype=np.float32) / self.S
        return torch.tensor(img), torch.tensor(box)

# 2. Model
class LocNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*(self.S//4)**2, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # outputs x1,y1,x2,y2
            nn.Sigmoid()        # ensure [0,1]
        )
    def forward(self, x):
        return self.fc(self.conv(x))

# 3. Training loop
dataset = SquareDataset()
loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = LocNet()
criterion = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    total_loss = 0
    for imgs, boxes in loader:
        opt.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, boxes)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
```

Key takeaways

- Sigmoid at output ensures predictions lie in [0,1].
- MSELoss directly implements the loss from section 3.

### 5. Visualization / Geometry

Imagine the feature map as a spatial heatmap. The network learns correlations between where activations concentrate and the true box corners.

Visualizing predictions:

1. Upscale the 4 output numbers to pixel space by multiplying by image size.
2. Overlay ground truth box (green) vs predicted box (red) on the image.
    
    ```
          ┌───────────────────────────┐
          │ ■■■■■■■■■■■■■■■■■■■■■   │
          │ ■                       ■ │
          │ ■   Ground Truth Box   ■ │
          │ ■      (green)         ■ │
          │ ■         ┌────┐       ■ │
          │ ■         │pred│       ■ │
          │ ■         └────┘       ■ │
          │ ■                       ■ │
          │ ■■■■■■■■■■■■■■■■■■■■■   │
          └───────────────────────────┘
    ```
    

This contrast highlights the geometric error directly.

### 6. Common pitfalls & tips

- Forgetting to normalize coordinates → unstable training.
- Using raw pixel values vs floats in [0,1] → large gradient magnitudes.
- Mismatch in coordinate convention (center‐format vs corner‐format).
- Overweighting localization loss vs classification loss in joint objectives.
- Ignoring aspect‐ratio distortions when resizing images.

### 7. Interview-ready insights

- Explain why smooth L1 (Huber) loss often outperforms MSE for bounding-box regression.
- Compute IoU:
    
    ```python
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
        return inter / (areaA + areaB - inter)
    ```
    
- Relate single-object localization to multi-object detection via anchor boxes (YOLO, SSD) or region proposals (Faster R-CNN).
- Discuss trade-offs: dense anchor grid vs anchor-free keypoint methods (CornerNet, CenterNet).

### 8. Practice exercises

1. **Coordinate conversion**
    - Write functions to convert between (x1,y1,x2,y2) and (cx,cy,w,h).
    - Hint: `cx = (x1+x2)/2; w = x2-x1`.
2. **Implement smooth L1 loss**
    - Use the code in section 3, vectorized with NumPy.
    - Test on random predictions vs targets.
3. **Extend the PyTorch example**
    - Add a small classification head predicting “square present” vs “none.”
    - Use a combined loss: `L = L_cls + α * L_loc_smooth`.
    - Experiment with α to see its effect on box accuracy.
4. **Visualization script**
    - Given a trained model, plot 5 test images with ground truth and predicted boxes.
    - Use `matplotlib` rectangles and ensure axis turned off.
5. **IoU thresholding**
    - For a batch of predictions, count how many have IoU ≥ 0.5 with the ground truth.
    - Compute the localization accuracy metric.

---

## Landmark detection

### 1. Direct definition

Landmark detection locates predefined keypoints (“landmarks”) on an object—most commonly human facial features such as eyes, nose, mouth corners, and jawline—by predicting each landmark’s (x, y) coordinates within the image.

### 2. Concept intuition

Landmark detection refines object localization by pinpointing semantic points rather than just bounding boxes.

- Instead of “there’s a face here,” you answer “the left eye is at (x₁,y₁), the right eye at (x₂,y₂), ….”
- This enables face alignment, emotion analysis, gaze estimation, AR filters, and medical image annotation.

Why it matters:

- High‐precision applications (e.g., virtual try-on, driver-monitoring) rely on sub-pixel accuracy of keypoints.
- It’s a diagnostic tool in healthcare (e.g., detecting facial palsy by asymmetry of landmarks).

Prerequisites & refreshers

- Regression fundamentals (predicting real-valued outputs).
- Image coordinate system and normalization to [0, 1].
- Concept of heatmaps as spatial probability distributions.

### 3. Mathematical breakdown

## Direct coordinate regression

For N landmarks, network outputs a vector of length 2N:

```python
preds = [x1, y1, x2, y2, …, xN, yN]  # all normalized to [0,1]
targets = [x1*, y1*, x2*, y2*, …, xN*, yN*]
```

Use Mean Squared Error summed over all keypoints:

```python
loss = sum((p - t)**2 for p, t in zip(preds, targets)) / (2*N)
```

Variables

- xi, yi: predicted normalized coordinates for landmark i
- xi*, yi*: ground-truth normalized coordinates
- Division by (2 N) averages loss per coordinate.

## Heatmap‐based detection

Instead of regressing coords directly, output N heatmaps of size H × W. Each heatmap is a 2D Gaussian centered at the true landmark.

Loss per heatmap (pixel‐wise MSE):

```python
loss = sum((H_pred[i][u,v] - H_true[i][u,v])**2
           for i in range(N)
           for u in range(H)
           for v in range(W)) / (N*H*W)
```

Variables

- H_pred[i]: predicted heatmap for landmark i
- H_true[i]: ground-truth Gaussian heatmap
- (u,v): pixel indices.

Why heatmaps help

- Model learns spatial context and uncertainty.
- Backpropagated gradients are more informative around the landmark region.

### 4. Code & practical application

Below is a PyTorch example for direct coordinate regression on a small “five‐point” face dataset (you can substitute any dataset).

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import pandas as pd

# 1. Dataset wrapper: CSV contains image_path, x1,y1,...,x5,y5 normalized
class FaceLandmarks(Dataset):
    def __init__(self, csv_file, img_folder, transform=None):
        self.df = pd.read_csv(csv_file)
        self.folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(f"{self.folder}/{row['image_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128,128)).astype(np.float32)/255.0
        img = np.transpose(img, (2,0,1))  # C,H,W
        coords = row[['x1','y1','x2','y2','x3','y3','x4','y4','x5','y5']].values.astype(np.float32)
        return torch.tensor(img), torch.tensor(coords)

# 2. Simple CNN regressor
class LandmarkNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*(16*16), 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 5 landmarks × (x,y)
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)

# 3. Training loop
dataset = FaceLandmarks("landmarks.csv", "images/")
loader = DataLoader(dataset, batch_size=16, shuffle=True)
model = LandmarkNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    epoch_loss = 0
    for imgs, coords in loader:
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, coords)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1} – Loss: {epoch_loss/len(loader):.4f}")
```

Key practical notes

- Normalize image pixels and coordinates to [0, 1].
- Batch size and learning rate may require tuning.
- Data augmentation (rotation, scale) significantly improves robustness.

### 5. Visualization / Geometry

Overlay predicted landmarks on an image:

```python
import matplotlib.pyplot as plt
img, coords = dataset[0]
pred = model(img.unsqueeze(0)).detach().numpy().reshape(-1,2)
coords_true = coords.numpy().reshape(-1,2)
img_np = np.transpose(img.numpy(), (1,2,0))

plt.imshow(img_np)
plt.scatter(pred[:,0]*128, pred[:,1]*128, c='r', label='pred')
plt.scatter(coords_true[:,0]*128, coords_true[:,1]*128, c='g', label='true')
plt.legend()
plt.axis('off')
plt.show()
```

Geometric interpretation

- Each (x,y) is a point on the image grid.
- Discrepancy between red vs green highlights localization error.

### 6. Common pitfalls & tips

- Forgetting coordinate normalization → slow or unstable convergence.
- Ignoring aspect‐ratio when resizing images → skewed landmark positions.
- Direct regression struggles when landmark visibility varies; consider visibility flags.
- Overfitting to landmark-heavy regions; use dropout or weight decay.
- Heatmap‐based methods require careful Gaussian sigma choice (too wide blurs precision; too narrow can vanish).

### 7. Interview-ready insights

- Compare direct regression vs heatmap regression: trade-off between simplicity and spatial richness.
- Explain normalized mean error (NME):
    
    ```python
    NME = (1/N) * sum( ||pred_i - true_i||₂ ) / inter_ocular_distance
    ```
    
- Discuss how multi‐task learning (landmark + eyeblink or pose estimation) can regularize and improve accuracy.
- Mention state-of-the-art approaches: Hourglass networks, Cascaded CNNs, and transformer-based keypoint detectors.

### 8. Practice exercises

1. Coordinate conversion
    - Write functions to switch between normalized and pixel coordinates for landmark points.
2. Heatmap generation
    - Create ground‐truth heatmaps for each landmark with a 2D Gaussian centered at the true point.
3. Implement heatmap‐based network
    - Modify `LandmarkNet` to output 5 heatmaps (size 32×32) and decode peaks to (x,y).
4. Compute NME metric
    - Given batches of preds and truths, implement normalized mean error using eye-corner distance.
5. Data augmentation impact
    - Train model with/without random rotations and report changes in NME over a validation set.

---

## Object detection

### 1. Direct definition

Object detection extends object localization by not only drawing a bounding box around each object in an image but also predicting its class label. A detector outputs, for each object instance,

- a class probability vector (e.g., “dog” vs “cat” vs “background”)
- a bounding‐box (x₁, y₁, x₂, y₂)

### 2. Concept intuition

- Classification tells you *what’s* in the image.
- Localization tells you *where one* object is.
- Detection tells you *what and where* for *every* object instance.

Why it matters

- Self-driving cars: locate cars, pedestrians, traffic signs simultaneously.
- Retail analytics: count and classify items on shelves.
- Wildlife monitoring: detect animals in camera-trap images.

Detectors must balance:

- Accuracy of box & class
- Speed (real-time constraints)
- Handling multiple, overlapping instances

### 3. Mathematical breakdown

### Multi-task loss

A common design splits into two heads: classification and bounding-box regression. Overall loss for one image:

```python
L = (1/Ncls) * sum(L_cls(p_i, c_i*))  \
    + λ * (1/Nloc) * sum(L_loc(b_i, b_i*))
```

- Ncls = number of anchor/proposal locations
- Nloc = number of *positive* anchors (those assigned to a ground-truth object)
- p_i = predicted class‐score vector at location i
- c_i* = ground-truth class label at i (0 = background)
- b_i = predicted box parameters (e.g., tx, ty, tw, th)
- b_i* = transformed ground-truth box for anchor i
- λ = weight balancing the two losses

### Classification loss

Cross-entropy (for two-stage methods) or focal loss (to handle class imbalance):

```python
L_cls = – sum over classes [c_i* · log p_i]
```

### Localization loss

Smooth L₁ (Huber) on encoded offsets:

```python
# anchor box (x_a,y_a,w_a,h_a), true box (x,y,w,h)
t_x = (x – x_a) / w_a
t_y = (y – y_a) / h_a
t_w = log(w / w_a)
t_h = log(h / h_a)

L_loc = sum smooth_L1(pred_t – true_t)  over (t_x, t_y, t_w, t_h)
```

### 4. Code & practical application

Below is a **PyTorch** example training a tiny single-stage detector on a synthetic “shapes” dataset. Each image contains either a red square or a blue circle. The model must classify which shape is present and regress its bounding box.

```python
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import random

# 1. Synthetic dataset
class ShapeDataset(Dataset):
    def __init__(self, n=1000, img_size=64, shape_size=16):
        self.n, self.S, self.s = n, img_size, shape_size

    def __len__(self): return self.n

    def __getitem__(self, idx):
        img = np.zeros((self.S, self.S, 3), dtype=np.uint8)
        cls = random.choice([0,1])  # 0=square, 1=circle
        x = random.randint(0, self.S-self.s)
        y = random.randint(0, self.S-self.s)

        if cls==0:
            cv2.rectangle(img, (x,y), (x+self.s,y+self.s), (0,0,255), -1)
        else:
            cv2.circle(img, (x+self.s//2,y+self.s//2), self.s//2, (255,0,0), -1)

        # normalize image
        img = img.astype(np.float32)/255.0
        img = np.transpose(img, (2,0,1))
        # normalized box corners
        box = np.array([x, y, x+self.s, y+self.s], dtype=np.float32) / self.S

        return torch.tensor(img), torch.tensor(cls), torch.tensor(box)

# 2. Detector model: shared conv → two heads
class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.cls_head = nn.Linear(64, 2)   # two classes
        self.loc_head = nn.Sequential(
            nn.Linear(64, 4),
            nn.Sigmoid()                     # [0,1] range
        )

    def forward(self, x):
        f = self.backbone(x).view(x.size(0), -1)
        return self.cls_head(f), self.loc_head(f)

# 3. Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = ShapeDataset(n=2000)
loader = DataLoader(ds, batch_size=32, shuffle=True)
model = TinyDetector().to(device)
cls_loss = nn.CrossEntropyLoss()
loc_loss = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=1e-3)
lambda_loc = 5.0

for epoch in range(10):
    total = {"cls":0, "loc":0}
    for imgs, labels, boxes in loader:
        imgs, labels, boxes = imgs.to(device), labels.to(device), boxes.to(device)
        p_cls, p_box = model(imgs)

        loss_c = cls_loss(p_cls, labels)
        loss_l = loc_loss(p_box, boxes)
        loss = loss_c + lambda_loc * loss_l

        opt.zero_grad()
        loss.backward()
        opt.step()
        total["cls"] += loss_c.item()
        total["loc"] += loss_l.item()

    print(f"Epoch {epoch+1} | cls: {total['cls']/len(loader):.3f} "
          f"| loc: {total['loc']/len(loader):.3f}")
```

Key points

- Two‐head design shares features.
- Cross-entropy for shape classification; MSE for box regression.
- `λ` (here 5.0) balances the losses on different scales.

### 5. Visualization / Geometry

Overlay predictions on test images:

```python
import matplotlib.pyplot as plt
model.eval()
img, label, box_true = ds[0]
with torch.no_grad():
    p_cls, p_box = model(img.unsqueeze(0).to(device))
box_pred = p_box.cpu().numpy()[0] * ds.S

# plot
ax = plt.gca()
plt.imshow(np.transpose(img.numpy(), (1,2,0)))
# true box (green)
x1,y1,x2,y2 = box_true.numpy()*ds.S
ax.add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1,
                           edgecolor='g', lw=2, fill=False))
# pred box (red)
x1p,y1p,x2p,y2p = box_pred
ax.add_patch(plt.Rectangle((x1p,y1p), x2p-x1p, y2p-y1p,
                           edgecolor='r', lw=2, fill=False))
plt.axis('off')
plt.show()
```

Geometric insight

- The network learns to map global features to both “what” and “where.”
- Balancing λ shapes the error surface: too large → boxes perfect but classes wrong; too small → vice versa.

### 6. Common pitfalls & tips

- **Imbalanced classes**: Many backgrounds vs few objects → use focal loss or hard negative mining.
- **Loss scaling**: Poor λ choice leads to one head dominating gradients.
- **Overlapping detections**: Requires Non-Maximum Suppression (NMS) to prune duplicates.
- **Anchors & aspect-ratios**: Hand-designed anchor sizes may miss unusual object shapes.
- **Input resizing**: Distorts objects if width≠height; maintain aspect ratio or pad.

### 7. Interview-ready insights

- Explain trade-offs between **two-stage** (Faster R-CNN) vs **one-stage** (YOLO, SSD) detectors: accuracy vs speed.
- Describe **anchor-free** methods (CenterNet, CornerNet) that predict keypoints instead of anchor offsets.
- Define **mean Average Precision (mAP)**: integrate precision–recall curve per class, then average.
- Detail NMS algorithm and its variants (soft-NMS).
- Discuss **feature pyramids** (FPN) for detecting multi-scale objects in one pass.

### 8. Practice exercises

1. Implement Non-Maximum Suppression (NMS) in NumPy:
    - Input: list of boxes + scores; Output: filtered boxes above an IoU threshold.
2. Augment the synthetic dataset with two shapes per image:
    - Modify the model to output two boxes and two class predictions; assign boxes via Hungarian matching or simple IoU assignment.
3. Integrate **focal loss** for classification:
    - Implement the focal loss forward pass; train detector and compare mAP.
4. Experiment with **λ** values:
    - Sweep λ∈[1,10,50] and plot classification accuracy vs. box‐regression RMSE.
5. Run a pretrained detector (e.g., `fasterrcnn_resnet50_fpn`) on your own photos using `torchvision` and visualize top-5 detections.

---

## Convolutional implementation of sliding windows

### 1. Direct definition

The convolutional implementation of sliding windows reformulates a classic “scan every subwindow with a classifier” approach as a single (or few) convolutional layers. Instead of explicitly extracting overlapping patches and running a neural network on each, you embed the classifier’s weights as convolutional filters that slide over the whole image in one pass, producing a 2D map of scores (and bounding‐box offsets).

### 2. Concept intuition

- Traditional sliding‐window detection: for each location and scale, crop a patch, forward it through your CNN/classifier, collect scores → **very** slow due to redundant computation.
- Convolutional trick: because convolution is exactly “dot‐product + bias” applied at every spatial location, you can take your patch‐based classifier’s final‐layer weights and use them as 1×1 (or k×k) conv filters.
- You thus share computation: low‐level feature maps are computed once, then the classifier filters convolve over these maps to produce a heatmap of detection scores.

Why it matters

- Cuts inference time by orders of magnitude.
- Lays the groundwork for single‐stage detectors (e.g., YOLO, SSD).
- Introduces the idea of **fully convolutional networks** that can handle arbitrary image sizes and output dense predictions.

### 3. Mathematical breakdown

Let’s say you originally had a patch classifier:

- Input: patch of shape (C, f, f)
- Fully connected (FC) layer with weights W_fc of shape (K, C·f·f) and bias b of shape (K,)
- Output: score vector of length K (e.g., K classes or K = 1 for binary detection)

Sliding‐window on full image of size (C, H, W) requires cropping every f×f region, flattening, then

```python
score_at_xy = W_fc · patch_vector + b
```

Convolutional re-interpretation:

1. Reshape W_fc to W_conv of shape (K, C, f, f)
2. Apply conv2d:output = conv2d(input, W_conv, bias=b, stride=s, padding=p)
3. output has shape (K, H', W'), where
    
    ```python
    H' = floor((H + 2*p - f) / s) + 1
    W' = floor((W + 2*p - f) / s) + 1
    ```
    

Each spatial cell (i,j) in that output exactly equals the dot‐product of the classifier on the f×f patch centered (or anchored) at the corresponding input location.

### 4. Code & practical application

Below is a PyTorch mini‐example turning a patch classifier into a conv detector. We use a toy feature extractor + patch classifier, then show the conv version.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Patch-based classifier (for 32×32 crops)
class PatchClassifier(nn.Module):
    def __init__(self, in_channels=3, f=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # output (32,1,1)
        )
        # FC layer classifying the single-patch feature
        self.fc = nn.Linear(32, 1)

    def forward(self, x_patch):
        # x_patch: (B,3,32,32)
        feat = self.features(x_patch)   # -> (B,32,1,1)
        feat = feat.view(feat.size(0), -1)  # -> (B,32)
        score = self.fc(feat)           # -> (B,1)
        return score

# 2. Convert to convolutional detector
class ConvDetector(nn.Module):
    def __init__(self, patch_model):
        super().__init__()
        # reuse feature extractor
        self.features = patch_model.features
        # take fc weights & reshape into conv filters
        W, b = patch_model.fc.weight.data, patch_model.fc.bias.data
        # W of shape (1, 32), reshape to (1,32,1,1)
        self.conv_score = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=1, bias=True
        )
        self.conv_score.weight.data = W.view(1,32,1,1)
        self.conv_score.bias.data   = b

    def forward(self, x):
        # x: (B,3,H,W) arbitrary H,W ≥ 32
        feat_map = self.features(x)     # -> (B,32,H',W')
        score_map = self.conv_score(feat_map)  # -> (B,1,H',W')
        return score_map

# 3. Demo
patch_model = PatchClassifier()
conv_model  = ConvDetector(patch_model)

# Input a full image
img = torch.randn(1,3,128,128)  # batch size 1
out_map = conv_model(img)       # shape (1,1,32,32) if pooling halves twice
print("Score map shape:", out_map.shape)
```

Key takeaways

- `AdaptiveAvgPool2d(1)` makes your FC→conv conversion straightforward.
- You get a dense score map: each cell corresponds to one sliding window.

### 5. Visualization / Geometry

```
                 Input image: 128×128
   ┌─────────────────────────────────────────┐
   │                                         │
   │       [◼︎ patch classifier’s f×f]       │
   │                                         │
   └─────────────────────────────────────────┘
```

After two MaxPool2d(stride=2) layers, feature map spatial size is H'=W'=128/2/2=32

Convolution with kernel_size=1 produces a 32×32 heatmap:

Each cell (i,j) ← score for the original 32×32 patch centered at (i*4, j*4) in the input

ASCII diagram of receptive fields versus score map:

```
input pixels  → receptive field of feat_map(5,10)
 ┌───────────────┐
 │               │
 │   32×32 patch │
 │     ╔════╗     │  ← conv_score 1×1 on feat_map[5,10]
 │     ║  x ║     │     yields score_map[5,10]
 │     ╚════╝     │
 │               │
 └───────────────┘
```

### 6. Common pitfalls & tips

- **Mismatch f and pooling**: Ensure your original patch size matches the receptive field of your conv features.
- **Padding & stride**: Off‐by‐one errors in output size if padding or stride aren’t carefully set.
- **Channel ordering**: When converting FC→conv, confirm `in_channels` and weight reshaping align.
- **Scale invariance**: A single filter size only sees one scale; for multi‐scale detection, apply conv detector at image pyramid levels or use multiple filter sizes.
- **Score calibration**: Raw conv scores may need sigmoid or softmax activated per cell.

### 7. Interview-ready insights

- Explain how **fully convolutional networks** generalize classification nets to dense prediction tasks (segmentation, detection).
- Derive the **receptive field** of any output cell: product of kernel sizes and strides across layers; critical for matching patch classifiers.
- Discuss the evolution from sliding‐window conv detectors to **anchor‐based** (SSD) and **anchor‐free** (FCOS) methods, which still hinge on conv mapping.
- Describe speed vs accuracy trade‐offs: single conv pass over full image vs explicit cropping.

### 8. Practice exercises

1. Manual vs conv sliding window
    - Implement a small patch classifier over 64×64 random images by explicitly cropping 32×32 windows with stride=16, comparing scores to the conv implementation’s heatmap.
2. Compute receptive field
    - Given a stack of conv(3×3,p=1) + maxpool(2) layers, write a function to compute each layer’s receptive field size and center offset.
3. Multi-scale detection
    - Create an image pyramid (scales 1.0, 0.75, 0.5), run your conv detector on each, then combine score maps into one detection list (threshold + NMS).
4. Anchors via convolution
    - Extend your `ConvDetector` with a second conv head predicting 4 offsets per anchor per cell (e.g., 3 anchor ratios); practice reshaping conv outputs into (N_anchors × 4 × H' × W') and decoding boxes.
5. Visualize gradients
    - For a selected cell in the score map, backpropagate its score to the input image (guided backprop), and visualize which pixels most influenced that score.

---

## Bounding box prediction

### 1. Direct definition

Bounding-box prediction is the task of regressing the coordinates of object boxes—usually parameterized as

```python
(x_center, y_center, width, height)
```

—from image feature maps. A detection head outputs four real‐valued offsets per box (or per anchor/proposal) that, when decoded, become the final rectangle in image space.

### 2. Concept intuition

When you look at an object in an image, you subconsciously estimate its center and extent. Bounding-box prediction teaches a network to do the same: transform high-level visual features into precise geometric cues.

Why it matters:

- Accurate box regression improves detection quality (higher IoU → better mAP).
- Proper parameterization (center/wh vs corner coords) stabilizes learning.
- It underpins both anchor-based (SSD, Faster R-CNN) and anchor-free (FCOS, CenterNet) detectors.

### 3. Mathematical breakdown

### Anchor-based parameterization

Given an anchor box with center (xₐ, yₐ), width wₐ, height hₐ, and a ground-truth box (x, y, w, h), you encode targets as:

```python
t_x = (x - x_a) / w_a
t_y = (y - y_a) / h_a
t_w = log(w / w_a)
t_h = log(h / h_a)
```

The model predicts offsets (p_x, p_y, p_w, p_h). During inference, decode with:

```python
x_pred = p_x * w_a + x_a
y_pred = p_y * h_a + y_a
w_pred = exp(p_w) * w_a
h_pred = exp(p_h) * h_a
```

### Loss function

A common choice is the Smooth L1 (Huber) loss over these four offsets:

```python
def smooth_l1(x, beta=1.0):
    if abs(x) < beta:
        return 0.5 * x**2 / beta
    return abs(x) - 0.5 * beta

L_loc = (smooth_l1(p_x - t_x)
       + smooth_l1(p_y - t_y)
       + smooth_l1(p_w - t_w)
       + smooth_l1(p_h - t_h)) / 4.0
```

Averaging keeps the regression loss on a similar scale to classification losses.

### 4. Code & practical application

Below are Python utilities (NumPy and PyTorch) to encode/decode boxes and compute the localization loss.

```python
import numpy as np
import torch
import torch.nn as nn

# 1. NumPy encode/decode
def encode_boxes(anchors, gt_boxes):
    # anchors, gt_boxes: arrays of shape (N,4) with (xc,yc,w,h)
    xa, ya, wa, ha = anchors.T
    x, y, w, h       = gt_boxes.T
    tx = (x - xa) / wa
    ty = (y - ya) / ha
    tw = np.log(w / wa)
    th = np.log(h / ha)
    return np.vstack([tx, ty, tw, th]).T

def decode_boxes(anchors, preds):
    xa, ya, wa, ha = anchors.T
    px, py, pw, ph = preds.T
    xc = px * wa + xa
    yc = py * ha + ya
    w  = np.exp(pw) * wa
    h  = np.exp(ph) * ha
    return np.vstack([xc, yc, w, h]).T

# 2. PyTorch Smooth L1 loss
class LocalizationLoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, preds, targets):
        diff = torch.abs(preds - targets)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff**2 / self.beta,
            diff - 0.5 * self.beta
        )
        return loss.mean()

# Example usage
anchors = np.array([[50,50,20,20], [30,80,40,10]], dtype=np.float32)
gt      = np.array([[55,45,18,22], [28,82,38,12]], dtype=np.float32)
encoded = encode_boxes(anchors, gt)

preds = torch.tensor(encoded + np.random.randn(*encoded.shape)*0.1)
targets = torch.tensor(encoded)
loss_fn = LocalizationLoss(beta=1.0)
loss = loss_fn(preds, targets)
print("Loc loss:", loss.item())
```

### 5. Visualization / Geometry

1. Draw anchors (grey), ground-truth boxes (green), and decoded predictions (red) on the image.
2. Each anchor’s center is a grid cell; its shape gives the initial box.
3. Offsets shift and scale the anchor to match the object.

ASCII sketch:

```
 [Anchor]   [Shift right/down by tx,ty]
     ┌─────┐       ┌───────────┐
     │     │  →    │           │
     └─────┘       └───────────┘
  scale by exp(tw,th) to match width/height
```

### 6. Common pitfalls & tips

- Mismatched coordinate formats (corner vs center) cause large, erratic gradients.
- Forgetting to normalize widths/heights → unstable logs.
- Using raw pixel offsets without anchors leads to slow convergence.
- Ignoring background anchors: only compute loc loss on positive anchors.
- Treating all anchors equally: implement hard negative mining or focal loss.

### 7. Interview-ready insights

- Discuss alternatives to Smooth L1: IoU-based losses (IoU, GIoU, DIoU, CIoU) that directly optimize overlap.
- Explain why log scaling for width/height is critical (handles scale variance).
- Describe anchor-free heads: they predict distances to four box edges instead of anchor offsets.
- Relate localization loss weight (λ) tuning to detection mAP sensitivity.

### 8. Practice exercises

1. **Implement IoU and GIoU losses**
    - Write functions for IoU(box1,box2) and GIoU; compare them on random boxes.
2. **Anchor clustering**
    - Use K-means on your dataset’s ground-truth widths/heights to find optimal anchor shapes.
3. **Anchor-free prediction**
    - Modify your encode/decode to predict distances `(l, t, r, b)` from a center point; train on a synthetic dataset.
4. **Visual debugging**
    - Plot predicted offsets as quiver arrows on anchor centers to see shift directions.
5. **Loss ablation**
    - Train a tiny detector head with different loc losses (MSE, Smooth L1, IoU); plot convergence curves and final IoU.

---

## Intersection over union

### 1. Direct definition

Intersection over Union (IoU) is a metric that quantifies the overlap between two bounding boxes. It’s defined as the ratio of the area of their intersection to the area of their union:

```python
IoU = area(intersection) / area(union)
```

### 2. Concept intuition

IoU measures how well a predicted box aligns with a ground-truth box:

- IoU = 1.0 → perfect overlap
- IoU = 0.0 → no overlap
- In object detection, predictions with IoU ≥ 0.5 (or 0.75) are considered correct hits.

Why it matters:

- Standardizes evaluation across shapes and scales.
- Feeds into Non-Maximum Suppression (NMS) to prune duplicate detections.
- Forms the basis of advanced localization losses (GIoU, DIoU, CIoU).

### 3. Mathematical breakdown

Given two boxes A and B, each as `(x1, y1, x2, y2)` where

- `(x1, y1)` is top-left corner
- `(x2, y2)` is bottom-right corner

Compute intersection:

```python
xi1 = max(A.x1, B.x1)
yi1 = max(A.y1, B.y1)
xi2 = min(A.x2, B.x2)
yi2 = min(A.y2, B.y2)

inter_width  = max(0, xi2 - xi1)
inter_height = max(0, yi2 - yi1)
area_inter   = inter_width * inter_height
```

Compute union via individual areas:

```python
area_A = (A.x2 - A.x1) * (A.y2 - A.y1)
area_B = (B.x2 - B.x1) * (B.y2 - B.y1)
area_union = area_A + area_B - area_inter
```

Finally, IoU:

```python
IoU = area_inter / area_union
```

### 4. Code & practical application

### NumPy implementation

```python
import numpy as np

def iou_numpy(boxA, boxB):
    # boxA, boxB: arrays [x1,y1,x2,y2]
    xi1 = np.maximum(boxA[0], boxB[0])
    yi1 = np.maximum(boxA[1], boxB[1])
    xi2 = np.minimum(boxA[2], boxB[2])
    yi2 = np.minimum(boxA[3], boxB[3])

    inter_w = np.maximum(0, xi2 - xi1)
    inter_h = np.maximum(0, yi2 - yi1)
    area_inter = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    area_union = areaA + areaB - area_inter

    return area_inter / area_union

# Example
A = np.array([10, 20, 50, 60])
B = np.array([30, 40, 70, 80])
print("IoU:", iou_numpy(A, B))
```

### PyTorch batch version

```python
import torch

def iou_torch(boxes1, boxes2):
    # boxes1, boxes2: tensors of shape (N,4)
    xi1 = torch.max(boxes1[:,0], boxes2[:,0])
    yi1 = torch.max(boxes1[:,1], boxes2[:,1])
    xi2 = torch.min(boxes1[:,2], boxes2[:,2])
    yi2 = torch.min(boxes1[:,3], boxes2[:,3])

    inter_w = (xi2 - xi1).clamp(min=0)
    inter_h = (yi2 - yi1).clamp(min=0)
    area_inter = inter_w * inter_h

    area1 = (boxes1[:,2] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1])
    area2 = (boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1])
    area_union = area1 + area2 - area_inter

    return area_inter / area_union

# Batch example
boxes1 = torch.tensor([[10,20,50,60],[0,0,10,10]], dtype=torch.float32)
boxes2 = torch.tensor([[30,40,70,80],[5,5,15,15]], dtype=torch.float32)
print("Batch IoUs:", iou_torch(boxes1, boxes2))
```

### 5. Visualization / Geometry

Imagine two rectangles on an image:

A: ■■■■■■■■■■

■ ■

■ ■

■■■■■■■■■■

B: ■■■■■■■■

■ ■

■ ■

■■■■■■■■

Their overlapping region is the common ■ area. IoU = (overlap area) ÷ (total covered area).

### 6. Common pitfalls & tips

- Zero or negative widths/heights if `(x2 <= x1)` or `(y2 <= y1)`. Always ensure valid boxes.
- Division by zero when both boxes have zero area. Add a small epsilon to denominator if needed.
- Integer vs float division: convert to floats before dividing.
- Choosing an IoU threshold too high (e.g., 0.9) may dismiss valid detections; too low (0.3) may accept poor overlaps.
- Remember to clamp intersection dimensions at zero to avoid negative areas.

### 7. Interview-ready insights

- Explain how IoU drives Non-Maximum Suppression: sort by score, then remove boxes with IoU ≥ threshold.
- Discuss IoU-based losses:
    - GIoU adds a penalty for non-overlapping boxes by considering the smallest enclosing box.
    - DIoU and CIoU incorporate distance between box centers and aspect-ratio consistency.
- Describe how IoU threshold affects precision–recall curves and mAP calculation.
- Mention alternatives for rotated boxes or masks: Generalized IoU, Dice coefficient.

### 8. Practice exercises

1. Implement a vectorized IoU matrix: for `N` predictions vs `M` ground truths, compute an `(N×M)` IoU tensor without Python loops.
2. Build a simple NMS:
    - Input: list of boxes and scores.
    - Output: filtered boxes above an IoU threshold.
3. Visualize IoU heatmap:
    - Fix one box; sweep a second box across the image grid.
    - Plot IoU value as a 2D heatmap of center offsets.
4. Compare IoU vs GIoU on edge cases:
    - Write both functions.
    - Generate pairs with no overlap and compute both metrics.
5. Integrate IoU as a loss:
    - Implement IoU loss `L = 1 - IoU`.
    - Train a toy box regressor on synthetic data and plot convergence of MSE vs IoU loss.

---

## Non-max suppression

### 1. Direct definition

Non-Maximum Suppression (NMS) is an algorithm that filters a set of overlapping bounding-box proposals by keeping only the highest-scoring box among highly overlapping ones. It greedily selects the box with the highest confidence score, then removes any remaining boxes whose Intersection over Union (IoU) with that box exceeds a chosen threshold.

### 2. Concept intuition

When your detector proposes many boxes around the same object, you want to collapse them into a single prediction. NMS does this by:

- Picking the most confident box
- Eliminating boxes that overlap too much with it
- Repeating until no boxes remain

Why it matters

- Prevents duplicate detections of the same object
- Keeps output concise and interpretable
- Essential for real-time systems where downstream tasks (tracking, counting) require unique objects

### 3. Mathematical breakdown

Given:

- A list of N boxes [b₁, b₂, …, bₙ], each with coordinates (x1,y1,x2,y2)
- A confidence score sᵢ for each box
- An IoU threshold t (e.g., 0.5)

Algorithm steps:

```python
1. Sort all indices by descending score:
   idxs = argsort(scores)[::-1]

2. Keep an empty list: keep = []

3. While idxs is not empty:
     i = idxs[0]                     # index of current highest score
     keep.append(i)                  # select this box

     # Compute IoUs between box i and all other boxes in idxs[1:]
     ious = compute_iou(boxes[i], boxes[idxs[1:]])

     # Remove any j in idxs[1:] where ious[j] > t
     filtered = [j for j, io in zip(idxs[1:], ious) if io <= t]

     # Update idxs to only those remaining
     idxs = filtered

4. Return keep as the list of selected boxes.
```

Key variables

- idxs: current pool of candidate indices
- keep: final selected indices
- t: overlap threshold controlling aggressiveness

### 4. Code & practical application

### NumPy implementation

```python
import numpy as np

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    # boxes: (N,4) array of [x1,y1,x2,y2]
    # scores: (N,) array of confidences
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(scores)[::-1]

    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)

        # compute intersections
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        # compute IoU
        union = areas[i] + areas[idxs[1:]] - inter
        ious = inter / union

        # keep boxes with IoU <= threshold
        remaining = np.where(ious <= iou_threshold)[0]
        idxs = idxs[remaining + 1]

    return keep

# Example usage
boxes = np.array([[10,10,50,50], [12,12,52,52], [100,100,150,150]])
scores = np.array([0.9, 0.8, 0.75])
selected = non_max_suppression(boxes, scores, 0.5)
print("Kept indices:", selected)  # e.g., [0, 2]
```

### PyTorch batched NMS (using torchvision)

```python
import torch
from torchvision.ops import nms

# boxes: (N,4), scores: (N,)
boxes_t = torch.tensor(boxes, dtype=torch.float32)
scores_t = torch.tensor(scores, dtype=torch.float32)

keep_indices = nms(boxes_t, scores_t, iou_threshold=0.5)
print("Kept by torchvision NMS:", keep_indices.tolist())
```

### 5. Visualization / Geometry

Imagine three overlapping predictions on one object:

1. High-score box (red)
2. Slightly offset box (orange)
3. Distant box on a different object (green)

ASCII:

```
┌───────┐       ┌────┐
│  red  │       │green│
│       │       └────┘
└───────┘
  ┌─────────┐
  │  orange │
  └─────────┘
```

NMS will:

- Pick red
- Remove orange (IoU(red,orange) > t)
- Keep green

### 6. Common pitfalls & tips

- **Threshold choice**
    - Too low (e.g., 0.3) → may keep duplicates
    - Too high (e.g., 0.7) → may remove distinct nearby objects
- **Coordinate format**
    - Ensure boxes use consistent (x1,y1,x2,y2) ordering
- **Performance**
    - Standard NMS is O(N²); for many boxes use GPU-accelerated or approximate versions
- **Class-aware NMS**
    - Run NMS per class to avoid suppressing detections across different object categories
- **Soft-NMS**
    - Instead of discarding boxes, decay their scores by IoU, which can improve recall

### 7. Interview-ready insights

- Explain why Greedy NMS can fail on crowded scenes—introduce **Soft-NMS** and **Adaptive NMS** as solutions.
- Describe **class-agnostic** vs **class-specific** NMS.
- Discuss the computational trade-off between traditional NMS and learning-based alternatives (e.g., **Learning NMS**modules).
- Highlight how NMS integrates into one-stage detectors (YOLO applies NMS on the concatenated output).
- Explain how NMS threshold affects the **precision–recall** curve and final **mAP**.

### 8. Practice exercises

1. **Implement Soft-NMS**
    - Instead of removing boxes with IoU > t, multiply their scores by `exp(-iou²/σ)` and re-sort.
2. **Class-aware NMS**
    - Given boxes, scores, and class labels, apply NMS independently per class and merge results.
3. **Approximate NMS**
    - Write a version that first clusters boxes into spatial buckets (e.g., via grid) to reduce pairwise comparisons.
4. **Threshold sweep**
    - Run NMS with thresholds in `[0.3,0.5,0.7]` on a sample detection set; plot number of kept boxes vs threshold.
5. **Visual debug**
    - For a test image with many proposals, visualize boxes before and after NMS to see the filtering effect.

---

## Anchor boxes

### 1. Direct definition

Anchor boxes are a set of predefined bounding-box templates—each defined by a width and height—tiled uniformly over an image (or feature map). At each spatial location, a detector predicts offsets and confidences for each anchor, turning these templates into final object proposals.

### 2. Concept intuition

Anchors let your network handle objects of different scales and aspect ratios in one pass.

- Instead of learning to regress every possible box shape from scratch, you start from a small set of “typical” boxes.
- At each cell, you ask: “Should there be an object of this shape here?” and “How must I shift/scale it?”

Why it matters

- Enables dense, multi-scale object coverage without explicit image pyramids.
- Simplifies matching ground-truth boxes to prediction templates during training.
- Underpins SSD, Faster R-CNN, RetinaNet and many one-stage detectors.

### 3. Mathematical breakdown

## Anchor parameter formulas

Given

- a base anchor size `s` (e.g., 32 px),
- aspect ratios `r` (e.g., [0.5, 1, 2]),
- scales `k` (e.g., [1, 1.5, 2]),

compute each anchor’s width `w_a` and height `h_a` as:

```python
w_a = s * k * sqrt(r)
h_a = s * k / sqrt(r)
```

Place these at each feature-map cell (i,j) whose center in the original image is:

```python
x_ctr = (j + 0.5) * stride
y_ctr = (i + 0.5) * stride
```

Then convert center/size anchors to corner format `(x1,y1,x2,y2)` via:

```python
x1 = x_ctr - w_a/2;  y1 = y_ctr - h_a/2
x2 = x_ctr + w_a/2;  y2 = y_ctr + h_a/2
```

## Anchor clustering (optional)

To pick optimal anchor shapes, run k-means on your dataset’s ground-truth box widths/heights using IoU-based distance:

```python
# distance(box, centroid) = 1 - IoU(box, centroid_anchor)
```

### 4. Code & practical application

```python
import numpy as np

# 1. Generate anchor shapes (w, h) for one base size
def make_anchors(base_size, ratios, scales):
    anchors = []
    for r in ratios:
        w = base_size * np.sqrt(r)
        h = base_size / np.sqrt(r)
        for k in scales:
            anchors.append([w * k, h * k])
    return np.array(anchors, dtype=np.float32)  # shape (A,2)

# 2. Tile anchors over feature map
def grid_anchors(feat_h, feat_w, stride, anchor_shapes):
    # compute cell centers
    shift_x = (np.arange(feat_w) + 0.5) * stride
    shift_y = (np.arange(feat_h) + 0.5) * stride
    shifts = np.stack(np.meshgrid(shift_x, shift_y), axis=-1).reshape(-1,2)  # (S,2)

    A = anchor_shapes.shape[0]
    S = shifts.shape[0]
    # repeat centers and shapes
    centers = np.repeat(shifts, A, axis=0)          # (S*A,2)
    shapes  = np.tile(anchor_shapes, (S,1))         # (S*A,2)

    # convert to corner coords
    x_ctr, y_ctr = centers[:,0], centers[:,1]
    ws, hs       = shapes[:,0], shapes[:,1]
    x1 = x_ctr - ws/2;  y1 = y_ctr - hs/2
    x2 = x_ctr + ws/2;  y2 = y_ctr + hs/2

    return np.stack([x1,y1,x2,y2], axis=1)          # (S*A,4)

# Example usage
base_size = 32
ratios = [0.5, 1.0, 2.0]
scales = [1.0, 1.5, 2.0]
shapes = make_anchors(base_size, ratios, scales)    # (9,2)
all_anchors = grid_anchors(feat_h=20, feat_w=20,
                           stride=16, anchor_shapes=shapes)
print("Total anchors:", all_anchors.shape[0])       # 20*20*9 = 3600
```

Key points

- `feat_h, feat_w` come from your backbone’s final feature map size.
- `stride` is how much receptive fields move in image pixels per feature map cell.
- You now have a dense set of 3600 candidate boxes for a 20×20 map with 9 anchors each.

### 5. Visualization / Geometry

Think of anchors as a grid of rectangles over the image:

- Each row/column corresponds to a feature-map cell.
- Each cell has A rectangles of different shapes.

When you overlay them, you see multi-scale coverage:

```
┌────────────────────────────────┐
│[]  [ ]  [  ]  [ ]  [ ]  [ ]    │  anchors of ratio<1
│[  ] [  ][ ]   [  ] [  ]  [ ]   │  anchors of ratio=1
│[]  [ ]  [  ]  [ ]  [ ]  [ ]    │  anchors of ratio>1
└────────────────────────────────┘
```

Each bracket is one anchor; colors or opacity can encode scale.

### 6. Common pitfalls & tips

- Forgetting to clip anchors to image boundaries → negative coords or overflow.
- Choosing too many anchors → heavy memory and computation.
- Too few/poorly chosen aspect ratios → low recall for unusual objects.
- Mismatch in `stride` between feature map and anchor placement.
- Inconsistent unit conventions (pixel vs normalized coordinates) when training.

### 7. Interview-ready insights

- Explain how anchor matching works: assign a ground-truth box to the anchor with highest IoU, and to all anchors with IoU ≥ threshold (e.g., 0.7).
- Describe anchor-free alternatives (FCOS, CenterNet) that predict offsets from a point without templates.
- Discuss anchor clustering (as in YOLOv2) using k-means with IoU distance to find dataset-specific shapes.
- Highlight how `stride` and receptive-field size determine the smallest/largest object you can detect.
- Contrast one-stage (SSD, RetinaNet) vs two-stage (Faster R-CNN) treatments of anchors.

### 8. Practice exercises

1. Anchor clustering
    - Extract widths/heights of ground-truth boxes from a dataset; run k-means (with custom IoU distance) to find 5–9 optimal anchor shapes.
2. Anchor generator
    - Write a function that, given any backbone output size and stride, returns all anchors in `(x1,y1,x2,y2)` format and visualizes them on sample images.
3. Anchor matching
    - Implement the assignment logic: for a batch of anchors and ground-truths, compute IoUs, then label each anchor as positive, negative, or ignored based on two thresholds.
4. Coverage analysis
    - For your clustered anchors and a validation set, compute recall: percentage of gt boxes with at least one anchor IoU ≥ 0.5.
5. Memory/time profiling
    - Vary number of ratios and scales; measure the effect on the number of anchors, forward-pass time, and GPU memory usage.

---

## YOLO algorithm

### 1. Direct definition

YOLO (“You Only Look Once”) is a single-stage object detector that divides an image into an S×S grid and, for each cell, directly predicts B bounding boxes, confidence scores, and C class probabilities in one forward pass. Each grid cell is responsible for objects whose center falls within it, combining classification and localization into a unified regression problem. This end-to-end approach yields real-time detection performance.

### 2. Concept intuition

YOLO treats detection as a single regression problem from image pixels to bounding‐box coordinates and class probabilities. By processing the entire image at once, it learns global context and avoids redundant region proposals. Each grid cell learns to spot objects and estimate their shapes, trading some localization precision for extreme speed. This global reasoning also helps reduce false positives in background regions.

### 3. Mathematical breakdown

YOLO’s loss is a sum of squared errors over all grid cells, balancing coordinate, confidence, and classification losses:

```python
# loss per cell i, box j, class c
λ_coord = 5
λ_noobj = 0.5

# coords: x, y (relative to cell), w, h (sqrt)
loss_loc = λ_coord * Σ_{i=1..S²} Σ_{j=1..B}
    [ (x_ij - x̂_ij)² + (y_ij - ŷ_ij)²
    + (√w_ij - √ŵ_ij)² + (√h_ij - √ĥ_ij)² ]

# confidence: objectness score
loss_conf = Σ_{i,j} [ (C_ij - Ĉ_ij)² * 1_obj
                   + λ_noobj * (C_ij - Ĉ_ij)² * 1_noobj ]

# classification:
loss_cls = Σ_{i=1..S²} 1_obj(i) * Σ_{c=1..C} (p_ic - p̂_ic)²

total_loss = loss_loc + loss_conf + loss_cls
```

Variables

- $(x̂,ŷ,ŵ,ĥ): ground‐truth$
- $C: predicted objectness$
- $p̂_ic: ground‐truth class one‐hot$
- $1_obj(i): indicator if cell i contains an object$

### 4. Code & practical application

```python
import torch
import torch.nn as nn

class YOLOHead(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        # final conv outputs S×S×(B*5 + C)
        self.conv = nn.Conv2d(512, B*5 + C, kernel_size=1)

    def forward(self, x):
        # x: (B,512,H,W) after backbone+pool
        out = self.conv(x)                   # (B,B*5+C,H,W)
        out = out.permute(0,2,3,1)           # (B,H,W,B*5+C)
        out = out.reshape(-1, self.S, self.S, self.B*5 + self.C)
        return out

# Decode one cell prediction at (i,j):
def decode_cell(pred, i, j, S, img_size):
    x, y, w, h, conf = pred[:5]
    cx = (j + x) * img_size / S
    cy = (i + y) * img_size / S
    bw = (w**2) * img_size
    bh = (h**2) * img_size
    return [cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2, conf]
```

This head attaches to any backbone (e.g., Darknet), producing a tensor you decode per cell and apply NMS.

### 5. Visualization / Geometry

Imagine a 7×7 grid overlay on a 448×448 image. Each cell predicts two boxes:

┌─────────────────────────────┐

│ [ ] [ ] [ ] … [ ] [ ] [ ] │

│ [ ] [ ] [ ] [ ] [ ] [ ] │

│ … │

└─────────────────────────────┘

A cell at (i,j) with offset (x,y) maps to a box center at

```
(cx, cy) = ((j + x)*w_cell, (i + y)*h_cell)
```

Width and height are extracted by squaring predictions to ensure positivity. This grid geometry lets one shot capture all objects.

### 6. Common pitfalls & tips

- Using raw w,h leads to negative or unstable sizes; YOLO predicts √w, √h.
- Only one object per grid cell: small or overlapping objects may be missed.
- Squared‐error loss penalizes small misalignments excessively—later versions switch to log‐space or IoU‐based losses.
- Imbalanced loss terms (λ_coord, λ_noobj) must be tuned to prevent vanishing gradients for confidence.
- Class predictions share space with boxes—large C relative to B can skew training.

### 7. Interview-ready insights

- Compare YOLO v1’s speed vs Faster R-CNN’s two-stage precision; discuss why v2 adds anchor boxes and batch normalization.
- Explain anchor‐based grid design in YOLOv2/YOLOv3 and multi‐scale feature maps for small-object detection.
- Highlight how YOLOv4 introduces CSP modules and Mosaic augmentation for better generalization.
- Discuss IOU losses (GIoU, DIoU) integrated in YOLOv5 and YOLOv7 for tighter box regression.
- Explore deployment: converting YOLO to ONNX/TensorRT for edge inference under 10 ms.

### 8. Practice exercises

- Implement grid‐label encoding: convert Pascal VOC boxes into S×S×(B*5+C) tensor of targets.
- Train a YOLO v1 head on a simplified shapes dataset; visualize loss curves and decode sample predictions.
- Vary λ_coord and λ_noobj; plot classification accuracy vs box RMSE.
- Extend the head to predict anchor‐offsets: add K anchor shapes and adjust loss accordingly.
- Deploy your trained PyTorch YOLO model with TorchScript, measure FPS on a webcam stream.

---

## Semantic segmentation with U-nets

### 1. Direct definition

Semantic segmentation with U-Nets assigns each pixel in an image to a semantic class (e.g., road, sky, person) using an encoder–decoder convolutional architecture that features skip connections between matching resolution levels.

### 2. Concept intuition

The goal of semantic segmentation is to produce a dense, per-pixel classification mask rather than a single label or bounding box.

U-Net’s design achieves this by combining:

- **Encoder path**: a sequence of conv + pooling layers that capture high-level context and enlarge the receptive field.
- **Decoder path**: upsampling layers that recover spatial resolution.
- **Skip connections**: direct links from encoder feature maps to decoder layers at the same spatial scale, preserving fine spatial details lost during pooling.

Why it matters

- Pixel-perfect masks are critical in medical imaging (tumor delineation), autonomous driving (road/lane inference), and satellite imagery (land-use classification).
- Skip connections balance “what” (context) with “where” (localization), enabling sharp boundaries.

### 3. Mathematical breakdown

### Encoder operations

At each level `ℓ`, for input feature map `Xᵢ` of shape `(Cᵢ, Hᵢ, Wᵢ)`:

```python
# two conv layers per block
Z = Conv2d(Xᵢ, out_channels=Cᵢ₊₁, kernel_size=3, padding=1)
A = ReLU(Z)
Z2 = Conv2d(A, out_channels=Cᵢ₊₁, kernel_size=3, padding=1)
A2 = ReLU(Z2)
P = MaxPool2d(A2, kernel_size=2, stride=2)  # halves H, W
```

### Decoder operations

At decoder level `ℓ`, receive upsampled map `U` and skip feature `S` of same resolution:

```python
# 1. Upsample
U_up = ConvTranspose2d(U, out_channels=C_dec, kernel_size=2, stride=2)

# 2. Concatenate skip connection
M = Concat([U_up, S], dim=channel)  # doubles channels

# 3. Two conv layers to refine
Z = Conv2d(M, out_channels=C_dec, kernel_size=3, padding=1)
A = ReLU(Z)
Z2 = Conv2d(A, out_channels=C_dec, kernel_size=3, padding=1)
A2 = ReLU(Z2)
```

### Final classification

After the last decoder block, apply a `1×1` convolution to map to `K` classes (no padding):

```python
logits = Conv2d(A2, out_channels=K, kernel_size=1)
```

### Loss function

We treat segmentation as per-pixel classification. Using pixel-wise cross-entropy:

```python
# logits: (B, K, H, W), targets: (B, H, W) with values 0..K-1
loss = -1 * sum_over_b,h,w [
    log_softmax(logits[b, :, h, w])[ targets[b, h, w] ]
] / (B * H * W)
```

Variables

- `B`: batch size
- `H, W`: output height/width
- `K`: number of classes

### 4. Code & practical application

Below is a PyTorch U-Net implementation and a toy training loop on synthetic circle masks.

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2

# 1. U-Net blocks
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # encoder
        self.enc1 = DoubleConv(1, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        # bottleneck
        self.bottleneck = DoubleConv(256, 512)
        # decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        # final 1x1 conv
        self.classifier = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # encoder
        x1 = self.enc1(x)       # (B,64,H,W)
        p1 = self.pool(x1)      # (B,64,H/2,W/2)
        x2 = self.enc2(p1)      # (B,128,H/2,W/2)
        p2 = self.pool(x2)      # (B,128,H/4,W/4)
        x3 = self.enc3(p2)      # (B,256,H/4,W/4)
        p3 = self.pool(x3)      # (B,256,H/8,W/8)
        # bottleneck
        b = self.bottleneck(p3) # (B,512,H/8,W/8)
        # decoder
        u3 = self.up3(b)        # (B,256,H/4,W/4)
        c3 = torch.cat([u3, x3], dim=1)
        d3 = self.dec3(c3)      # (B,256,H/4,W/4)
        u2 = self.up2(d3)       # (B,128,H/2,W/2)
        c2 = torch.cat([u2, x2], dim=1)
        d2 = self.dec2(c2)      # (B,128,H/2,W/2)
        u1 = self.up1(d2)       # (B,64,H,W)
        c1 = torch.cat([u1, x1], dim=1)
        d1 = self.dec1(c1)      # (B,64,H,W)
        return self.classifier(d1)  # (B,K,H,W)

# 2. Synthetic circle dataset
class CircleDataset(Dataset):
    def __init__(self, n=500, img_size=128):
        self.n, self.S = n, img_size
    def __len__(self): return self.n
    def __getitem__(self, idx):
        img = np.zeros((self.S, self.S), dtype=np.uint8)
        mask = np.zeros_like(img)
        # random circle
        r = np.random.randint(10, 30)
        cx = np.random.randint(r, self.S-r)
        cy = np.random.randint(r, self.S-r)
        cv2.circle(img, (cx, cy), r, 255, -1)
        cv2.circle(mask, (cx, cy), r, 1, -1)
        img = img.astype(np.float32)/255.0
        return (torch.tensor(img[None]), torch.tensor(mask, dtype=torch.long))

# 3. Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = CircleDataset(n=1000)
loader = DataLoader(ds, batch_size=8, shuffle=True)
model = UNet(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(15):
    total_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)                   # (B,2,H,W)
        loss = criterion(logits, masks)        # pixel-wise CE
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
```

Key takeaways

- Use `ConvTranspose2d` for learnable upsampling.
- Skip connections (`torch.cat`) preserve spatial detail.
- Loss expects shape `(B, H, W)` integer masks.

### 5. Visualization / Geometry

1. **Feature map sizes**
    - Input 128×128 → after three pools → 16×16 at bottleneck → upsample back to 128×128.
2. **Skip path overlay**
    
    Encoder feature maps (high resolution, low semantics)
    
    ─────────▶ skip ─────────┐
    
    ▼
    
    Decoder maps (low resolution, high semantics) ──► concatenate ──► refine
    
3. **Boundary sharpness**
    - Skip connections inject edge information lost by pooling, yielding crisp masks.

### 6. Common pitfalls & tips

- **Shape mismatches**: ensure `ConvTranspose2d` output size matches encoder feature map for concatenation.
- **Class imbalance**: majority background pixels can dominate loss; consider weighting or using Dice/focal loss.
- **Overfitting** on small datasets: use augmentations (rotations, flips, elastic transforms).
- **Checkerboard artifacts** with naive transposed conv: prefer bilinear upsampling + `Conv2d` when artifacts appear.
- **Memory footprint**: U-Nets grow quickly in channels; reduce base channel count or use mixed precision.

### 7. Interview-ready insights

- Contrast U-Net with FCN: U-Net’s symmetric decoder and skip concat yields sharper masks.
- Describe U-Net++ and Attention U-Net: nested skip paths and attention gates refine feature fusion.
- Discuss loss functions: combine cross-entropy with Dice loss for better boundary handling:
    
    ```python
    dice = (2 * intersection + eps) / (pred_area + true_area + eps)
    L_dice = 1 - dice
    total_loss = CE_loss + α * L_dice
    ```
    
- Real-world deployment: sliding-window inference on large images vs patch-based batch inference.

### 8. Practice exercises

1. **Implement shape checker**
    - Pass a random tensor through your U-Net and assert shapes at each encoder/decoder stage.
2. **Data augmentation pipeline**
    - Add random elastic deformations and brightness jitter; measure impact on validation IoU.
3. **Dice loss**
    - Code a PyTorch `nn.Module` that computes Dice loss for multi-class masks; integrate into training.
4. **Visualization script**
    - For a batch of validation images, plot input, ground-truth mask, and predicted mask side by side using `matplotlib`.
5. **Transfer learning**
    - Replace the encoder path with a pre-trained ResNet­34 backbone; adapt skip connections and retrain on a small segmentation dataset.
6. **Metric computation**
    - Implement pixel accuracy, mean IoU, and per-class IoU for evaluating segmentation performance.

---

## Transpose convolution

### 1. Direct definition

Transpose convolution (often called “deconvolution” or “fractionally-strided convolution”) is a learnable upsampling operation that takes a lower-resolution feature map and produces a higher-resolution output by effectively reversing the spatial transformation of a regular convolution.

### 2. Concept intuition

- A standard convolution “slides” a kernel over an input to shrink or maintain spatial dimensions.
- Transpose convolution inserts learnable overlaps: it spreads each input pixel over a patch of output pixels, then sums overlapping contributions.
- This lets the network learn how to upsample, filling in details rather than using a fixed rule (like bilinear).

Why it matters

- Decoder paths in segmentation and generative models (e.g., U-Net, GAN generators) rely on transpose conv for trainable upsampling.
- It adapts learned filters to reconstruct spatial structure lost in downsampling.

### 3. Mathematical breakdown

Let

- I = input size,
- K = kernel size,
- S = stride,
- P = padding,
- O = output size.

Transpose convolution’s output dimension:

```python
O = (I - 1) * S - 2*P + K
```

Each input pixel at position (i,j) is multiplied by the K×K kernel and scattered into the output at stride-spaced locations. Overlaps from adjacent inputs sum up.

### 4. Code & practical application

### PyTorch example

```python
import torch
import torch.nn as nn

# input: batch of 1, 1 channel, 4×4
x = torch.arange(16, dtype=torch.float32).view(1,1,4,4)

# transpose conv: kernel 3×3, stride 2, padding 1
tconv = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1)
# initialize for demonstration
nn.init.constant_(tconv.weight, 1.0); nn.init.zeros_(tconv.bias)

y = tconv(x)
print("Input shape:", x.shape)
print("Output shape:", y.shape)
print(y[0,0])
```

### TensorFlow/Keras example

```python
from tensorflow.keras.layers import Conv2DTranspose
import numpy as np

x_np = np.arange(16).reshape(1,4,4,1).astype('float32')
layer = Conv2DTranspose(1, 3, strides=2, padding='same',
                        kernel_initializer='ones', use_bias=False)
y_np = layer(x_np)
print("Output shape:", y_np.shape)
```

### 5. Visualization / Geometry

1. Start with a 2×2 input and a 3×3 kernel.
2. For stride=2, imagine inserting one zero row/column between every input row/column, creating a 3×3 “stretched” grid.
3. Convolving the kernel over that stretched grid yields overlapping patches that sum to form the final (5×5) output.

This overlap is what gives transpose convolution its learned interpolation behavior.

### 6. Common pitfalls & tips

- Checkerboard artifacts: uneven overlap can produce “ripples.” Mitigate by using even kernel sizes or pairing with fixed upsampling (e.g., bilinear) + Conv2D.
- Misaligned output size: off-by-one errors if padding/stride mismatched. Compute `O` with the formula first.
- Initialization matters: all-ones weight exaggerates overlaps. Use standard initializers (He/Xavier).
- Slow on large feature maps: consider upsampling + regular conv for efficiency.

### 7. Interview-ready insights

- Contrast transpose conv vs. nearest‐neighbor or bilinear upsampling followed by Conv2D: the latter avoids checkerboard but is two operations.
- Explain how overlapping kernel placements cause learned interpolation, versus fixed interpolation schemes.
- Derive the transpose convolution output shape formula and discuss how padding in the “inverse” direction differs.
- Mention alternatives: sub-pixel convolution (PixelShuffle) rearranges channels to spatial dims and avoids overlap issues.

### 8. Practice exercises

1. **Manual 1D transpose convolution**
    - Implement a 1D transpose conv over a small array with stride=2, kernel=[1,2,1], and verify output by hand.
2. **Output size sanity check**
    - Write a function that, given `(I, K, S, P)`, returns expected `O` and tests various parameter combos.
3. **Upsample+Conv vs TransposeConv**
    - Build two Keras models: one with `UpSampling2D`+`Conv2D`, another with `Conv2DTranspose`. Compare output feature maps and training speed on a toy segmentation task.
4. **Artifact reduction**
    - Train a small decoder with transpose conv on a toy dataset of shapes. Visualize results and replace transpose conv with bilinear upsampling + Conv2D to compare checkerboard artifacts.
5. **PixelShuffle alternative**
    - Implement a sub-pixel convolution layer in PyTorch using `nn.PixelShuffle` and show how it rearranges a channel-stacked tensor into a higher-res feature map.

---

## U-Net architecture

### 1. Direct definition

The U-Net architecture is a symmetric encoder–decoder convolutional network for semantic segmentation. It consists of:

- A **contracting path** (encoder) that captures context via successive Conv→ReLU→Conv→ReLU→MaxPool blocks,
- A **bottleneck** layer connecting encoder and decoder,
- An **expanding path** (decoder) that restores spatial resolution via ConvTranspose (upsampling) layers and concatenates corresponding encoder feature maps (“skip connections”), followed by Conv→ReLU→Conv→ReLU blocks.

### 2. Concept intuition

- **“U” shape**: encoder on the left, decoder on the right, skip links bridging matching resolutions.
- **Encoder** extracts increasingly abstract features while halving H×W at each level and doubling channels.
- **Decoder** upsamples feature maps, halves channels, and uses skip connections to recover precise localization lost during pooling.
- **Skip connections** inject fine-grained details into the coarse, semantic decoder features—crucial for sharp boundaries.

Why it matters

- Balances “what” (deep, semantic features) with “where” (shallow, spatial features).
- Lightweight and effective on small datasets (medical scans, satellite imagery).
- Forms a blueprint for countless segmentation variants (Attention U-Net, U-Net++, Residual U-Net).

### 3. Mathematical breakdown

### Encoder block (at level ℓ)

Input feature map: (X^ℓ) of shape ((C^ℓ, H^ℓ, W^ℓ))

1. Convolution
    
    ```python
    Z1 = Conv2d(X^ℓ, out_channels=C^{ℓ+1}, kernel_size=3, padding=1)
    A1 = ReLU(Z1)
    Z2 = Conv2d(A1, out_channels=C^{ℓ+1}, kernel_size=3, padding=1)
    A2 = ReLU(Z2)
    ```
    
2. Downsample
    
    ```python
    P^ℓ = MaxPool2d(A2, kernel_size=2, stride=2)
    # output shape: (C^{ℓ+1}, H^ℓ/2, W^ℓ/2)
    ```
    

### Bottleneck

Input: (P^L) at deepest level (L)

```python
Z1 = Conv2d(P^L,  out_channels=C^{L+1}, kernel_size=3, padding=1); A1 = ReLU(Z1)
Z2 = Conv2d(A1,   out_channels=C^{L+1}, kernel_size=3, padding=1); A2 = ReLU(Z2)
```

### Decoder block (at level ℓ, reversed order)

Input: upsampled (U^{ℓ+1}) and skip feature (A2^ℓ)

1. Upsample
    
    ```python
    U_up = ConvTranspose2d(U^{ℓ+1}, out_channels=C^{ℓ+1}, kernel_size=2, stride=2)
    # output shape matches A2^ℓ: (C^{ℓ+1}, H^ℓ, W^ℓ)
    ```
    
2. Concatenate
    
    ```python
    M^ℓ = Concat([U_up, A2^ℓ], dim=channel)
    # shape: (2*C^{ℓ+1}, H^ℓ, W^ℓ)
    ```
    
3. Two conv layers
    
    ```python
    Z1 = Conv2d(M^ℓ, out_channels=C^ℓ, kernel_size=3, padding=1); A1 = ReLU(Z1)
    Z2 = Conv2d(A1,  out_channels=C^ℓ, kernel_size=3, padding=1); A2 = ReLU(Z2)
    # output: (C^ℓ, H^ℓ, W^ℓ)
    ```
    

### Final classifier

After last decoder block:

```python
logits = Conv2d(A2^0, out_channels=K, kernel_size=1)
# K = number of classes; output shape: (K, H, W)
```

### 4. Code & practical application

A flexible PyTorch U-Net blueprint that adapts to arbitrary depths and channel sizes:

```python
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, depth=4, num_classes=2):
        super().__init__()
        self.depth = depth

        # build encoder
        enc_layers = []
        ch = in_channels
        for d in range(depth):
            enc_layers.append(self._double_conv(ch, base_channels * 2**d))
            ch = base_channels * 2**d
        self.encoders = nn.ModuleList(enc_layers)
        self.pool = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = self._double_conv(ch, base_channels * 2**depth)
        ch = base_channels * 2**depth

        # build decoder
        dec_ups  = []
        dec_convs= []
        for d in reversed(range(depth)):
            up_ch = base_channels * 2**d
            dec_ups.append(nn.ConvTranspose2d(ch, up_ch, kernel_size=2, stride=2))
            dec_convs.append(self._double_conv(ch, up_ch))
            ch = up_ch
        self.decoders_up = nn.ModuleList(dec_ups)
        self.decoders_conv = nn.ModuleList(dec_convs)

        # final classifier
        self.classifier = nn.Conv2d(ch, num_classes, kernel_size=1)

    def _double_conv(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, conv, skip in zip(self.decoders_up, self.decoders_conv, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = conv(x)

        return self.classifier(x)
```

Key notes

- `base_channels` controls model width; doubling at each encoder depth.
- `depth` sets how many pooling/upsampling layers.
- Skip connections stored in `skips` list and used in reverse order.

### 5. Visualization / Geometry

```
Input (H×W)
    │
  [64, H, W] ← DoubleConv
    ↓ pool
  [64, H/2, W/2] ← DoubleConv
    ↓ pool
  [128, H/4, W/4]
    ↓ pool
    ...
    ↓ pool
  [512, H/16, W/16]  ← bottleneck
    ↑ upconv
  [256, H/8,  W/8 ] ← concat skip from encoder level 3
    ↑ upconv
  [128, H/4,  W/4 ] ← concat skip from encoder level 2
    ↑ upconv
  [ 64, H/2,  W/2 ] ← concat skip from encoder level 1
    ↑ upconv
  [ 64, H,    W   ] ← concat skip from encoder level 0
```

Each arrow shows a 2× upsampling (ConvTranspose) and concatenation with the feature map of the same spatial size from the encoder.

### 6. Common pitfalls & tips

- **Shape mismatches**: Transposed conv and pooling must mirror each other so skip tensors match in H×W.
- **Checkerboard artifacts**: If you see grid-like noise, consider using `Upsample(mode='bilinear')` + `Conv2d` instead of `ConvTranspose2d`.
- **Overfitting**: U-Nets can memorize small datasets; mitigate with data augmentation (flips, rotations, elastic deformations).
- **Memory footprint**: Deep U-Nets with high `base_channels` may exceed GPU memory; adjust depth or use mixed precision.
- **BatchNorm placement**: Inserting `BatchNorm2d` after each Conv can improve training stability.

### 7. Interview-ready insights

- **Receptive field**: Calculate how many input pixels influence one output pixel by tracing kernel sizes and strides through encoder + decoder.
- **Skip connection rationale**: Explain why concatenating encoder features preserves spatial accuracy and combats vanishing gradients.
- **Variants**:
    - **U-Net++** adds nested dense skip paths to gradually refine features across scales.
    - **Attention U-Net** gates skip features with attention to suppress irrelevant activations.
    - **Residual U-Net** replaces double-conv blocks with residual bottleneck blocks for deeper backbones.
- **Loss functions**: Combine cross-entropy with Dice or focal loss to handle class imbalance in segmentation tasks.

### 8. Practice exercises

1. **Dynamic U-Net**
    - Modify the blueprint to accept any input size (not multiples of 2ᴰ) by computing correct padding and output paddings.
2. **Receptive field calculator**
    - Write a script that, given your `UNet` model, computes the receptive field of each layer and the overall network.
3. **Hybrid upsampling**
    - Replace all `ConvTranspose2d` layers with `nn.Upsample(mode='bilinear', scale_factor=2)` followed by `Conv2d` blocks. Compare qualitative outputs on a toy dataset.
4. **Attention gates**
    - Implement a simple attention gate module that weighs skip features before concatenation (as in Attention U-Net) and integrate it into your decoder.
5. **Performance profiling**
    - Vary `depth` and `base_channels`; measure GPU memory usage and per-batch inference time to find an optimal trade-off for your hardware.
6. **Transfer learning**
    - Swap the encoder path for a pretrained backbone (ResNet-34); adapt skip connections and finetune on a small segmentation set, observing convergence speed and accuracy.

---