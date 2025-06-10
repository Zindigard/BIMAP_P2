import os
import numpy as np
import matplotlib.pyplot as plt
from czifile import imread
from read_roi import read_roi_zip
from cellpose import models
from skimage.draw import polygon2mask
from sklearn.metrics import jaccard_score, f1_score, accuracy_score

# === Settings ===
czi_file = "WT_NADA_RADA_HADA_NHS_40min_ROI1_SIM.czi"
roi_zip = "RoiSet_Contour_bacteria_ROI1.zip"

# === Step 1: Load CZI image ===
print(f"📂 Opening image: {czi_file}")
raw = imread(czi_file)
print(f"✅ Raw CZI shape: {raw.shape}")

img = np.squeeze(raw)[0:3, :, :]  # Extract RGB channels
print(f"✅ Final image shape: {img.shape}")

# Normalize and re-order to RGB
def normalize(ch):
    return (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)

rgb = np.stack([
    normalize(img[2]),  # Blue → Red
    normalize(img[1]),  # Green
    normalize(img[0])   # Red → Blue
], axis=-1)
print(f"🖼️ RGB image shape: {rgb.shape}")

# === Step 2: Load ROI ground-truth masks ===
if not os.path.exists(roi_zip):
    print(f"⚠️ ROI zip not found: {roi_zip}")
    exit()

rois = read_roi_zip(roi_zip)
print(f"✅ Loaded {len(rois)} ROI(s)")

ground_truth_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)

for i, roi in enumerate(rois.values(), start=1):
    x, y = roi['x'], roi['y']
    mask = polygon2mask(rgb.shape[:2], np.column_stack((y, x)))
    ground_truth_mask[mask] = 1

# Create evaluation region: only where GT exists
eval_region_mask = ground_truth_mask > 0

# === Step 3: Cellpose segmentation ===
print("🧠 Segmenting RGB image with Cellpose...")
model = models.Cellpose(gpu=False, model_type='cyto')
masks, flows, styles, _ = model.eval(rgb, diameter=None, flow_threshold=0.4, channels=[0, 1])
print(f"✅ Segmentation complete. Detected {masks.max()} cells.")

# === Step 4: Binarize Cellpose output for metric comparison ===
segmented_mask = (masks > 0).astype(np.uint8)

# === Step 5: Evaluation metrics ===
def dice_score(gt, pred):
    intersection = np.logical_and(gt, pred).sum()
    return 2.0 * intersection / (gt.sum() + pred.sum() + 1e-8)

def calculate_metrics(gt, pred, eval_mask):
    gt_eval = gt[eval_mask]
    pred_eval = pred[eval_mask]
    gt_flat = gt_eval.flatten()
    pred_flat = pred_eval.flatten()
    jaccard = jaccard_score(gt_flat, pred_flat, average='binary')
    f1 = f1_score(gt_flat, pred_flat, average='binary')
    acc = accuracy_score(gt_flat, pred_flat)
    dice = dice_score(gt_eval, pred_eval)
    return jaccard, f1, dice, acc

jaccard, f1, dice, accuracy = calculate_metrics(ground_truth_mask, segmented_mask, eval_region_mask)

# === Step 6: Print results ===
print("\n📊 Evaluation Metrics (within GT area only):")
print(f"Jaccard Index : {jaccard:.4f}")
print(f"F1 Score      : {f1:.4f}")
print(f"Dice Score    : {dice:.4f}")
print(f"Accuracy      : {accuracy:.4f}")

# === Step 7: Visualization Panels ===
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(rgb)
axs[0].set_title("Original RGB Image")
axs[0].axis('off')

axs[1].imshow(ground_truth_mask, cmap='gray')
axs[1].set_title("Ground Truth Mask")
axs[1].axis('off')

axs[2].imshow(segmented_mask, cmap='gray')
axs[2].set_title("Cellpose Prediction")
axs[2].axis('off')

axs[3].imshow(eval_region_mask, cmap='Reds')
axs[3].set_title("Evaluation Region (GT Area)")
axs[3].axis('off')

plt.tight_layout()
plt.show()

# === Step 8: Overlay GT vs Prediction (Color Coded) ===
# Create RGB mask where:
# Green = GT only, Red = Prediction only, Yellow = Overlap

overlay = np.zeros((*rgb.shape[:2], 3), dtype=np.float32)

# Areas
gt_only = (ground_truth_mask == 1) & (segmented_mask == 0)
pred_only = (ground_truth_mask == 0) & (segmented_mask == 1)
overlap = (ground_truth_mask == 1) & (segmented_mask == 1)

# Apply colors
overlay[gt_only] = [0, 1, 0]     # Green
overlay[pred_only] = [1, 0, 0]   # Red
overlay[overlap] = [1, 1, 0]     # Yellow

plt.figure(figsize=(6, 6))
plt.imshow(overlay)
plt.title("Overlay: GT (Green), Prediction (Red), Overlap (Yellow)")
plt.axis('off')
plt.tight_layout()
plt.show()
