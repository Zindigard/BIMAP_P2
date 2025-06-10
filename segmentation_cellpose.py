import numpy as np
import matplotlib.pyplot as plt
from czifile import imread
from cellpose import models
from skimage import transform, measure, morphology

# --------- SETTINGS ---------
czi_path = "WT_NADA_RADA_HADA_NHS_40min_ROI1_SIM.czi"  # Path to the CZI file
# ----------------------------

# --- Load CZI image ---
print(f"📂 Opening image: {czi_path}")
raw = imread(czi_path)
print(f"✅ Raw CZI shape: {raw.shape}")

# Extract RGB channels (shape: 1,1,3,1,1,Y,X,1)
img = np.squeeze(raw)[0:3, :, :]  # (3, Y, X)
print(f"✅ Final image shape: {img.shape}")

# --- Normalize and re-order to RGB (swap R and B) ---
def normalize(ch):
    return (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)

rgb = np.stack([
    normalize(img[2]),  # Blue → Red
    normalize(img[1]),  # Green
    normalize(img[0])   # Red → Blue
], axis=-1)

print(f"🖼️ RGB image shape: {rgb.shape}")

# --- Cellpose Segmentation ---
print("🧠 Segmenting RGB image with Cellpose...")

# Create a Cellpose model
model = models.Cellpose(gpu=False, model_type='cyto')

# Adjust parameters to reduce over-segmentation
masks, flows, styles, diams = model.eval(
    rgb,
    diameter=None,  # Cellpose will try to estimate the diameter automatically
    flow_threshold=0.3,  # Increase to reduce over-segmentation
    cellprob_threshold=0.5,  # Increase to reduce false positives
    channels=[0, 0]  # We're using the full RGB channels for segmentation
)

print(f"✅ Segmentation complete. Detected {masks.max()} cells.")

# --- Post-processing to merge close cells ---
# Erosion followed by dilation to merge nearby cells
merged_masks = morphology.erosion(masks, np.ones((5, 5)))  # Erosion step
merged_masks = morphology.dilation(merged_masks, np.ones((5, 5)))  # Dilation step

# --- Create a color overlay from merged masks ---
def masks_to_rgb_overlay(base_img, masks):
    """Overlay random color masks on RGB image."""
    from skimage.color import label2rgb
    mask_rgb = label2rgb(masks, image=base_img, bg_label=0, kind='overlay')
    return mask_rgb

overlay = masks_to_rgb_overlay(rgb, merged_masks)

# --- Plotting ---
plt.figure(figsize=(10, 10))
plt.imshow(overlay)
plt.axis('off')
plt.title("🌈 RGB Cellpose Segmentation with Colored Cells")
plt.tight_layout()
plt.show()
