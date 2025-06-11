
import os
import numpy as np
from aicsimageio import AICSImage
from cellpose import models
from skimage.draw import polygon as sk_polygon
from read_roi import read_roi_file
import tifffile as tiff
import matplotlib.pyplot as plt

def polygon_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# Step 1: Load image to get actual shape
czi_path = "C:/FAU/Project/data sources/ROI1/WT_NADA_RADA_HADA_NHS_40min_ROI1_SIM.czi"
img = AICSImage(czi_path)
image_shape = img.get_image_data("YX", S=0, T=0, C=0).shape
img_shape = image_shape  # (height, width)

# Step 2: Create GT binary mask
roi_folder = "C:/FAU/Project/data sources/manually labeled/RoiSet_Contour_bacteria_ROI1"
roi_files = sorted([f for f in os.listdir(roi_folder) if f.endswith('.roi')])

gt_mask = np.zeros(img_shape, dtype=np.uint8)
gt_areas = []

for filename in roi_files:
    roi_data = read_roi_file(os.path.join(roi_folder, filename))
    roi = list(roi_data.values())[0]
    x = np.array(roi['x'])
    y = np.array(roi['y'])
    if len(x) < 3:
        continue
    rr, cc = sk_polygon(y, x, shape=gt_mask.shape)
    gt_mask[rr, cc] = 1
    area = polygon_area(x, y)
    gt_areas.append(area)

min_area = np.min(gt_areas)
max_area = np.max(gt_areas)

# Step 3: Segment image
rgb_image = np.stack([
    img.get_image_data("YX", S=0, T=0, C=0),
    img.get_image_data("YX", S=0, T=0, C=1),
    img.get_image_data("YX", S=0, T=0, C=2),
], axis=-1)

model = models.CellposeModel(gpu=False)
pred = model.eval(rgb_image, diameter=None)
masks = pred[0]

# Step 4: Build predicted mask with filtering by area
from skimage.measure import regionprops, label

filtered_pred_mask = np.zeros_like(masks, dtype=np.uint8)
labeled = label(masks)
regions = regionprops(labeled)

for r in regions:
    if min_area <= r.area <= max_area:
        coords = r.coords
        filtered_pred_mask[coords[:, 0], coords[:, 1]] = 1

# Step 5: Overlay masks for comparison
overlay = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
overlay[(gt_mask == 1) & (filtered_pred_mask == 0)] = [255, 0, 0]      # GT only - red
overlay[(gt_mask == 0) & (filtered_pred_mask == 1)] = [0, 255, 0]      # Pred only - green
overlay[(gt_mask == 1) & (filtered_pred_mask == 1)] = [255, 255, 0]    # Both - yellow

plt.figure(figsize=(10, 10))
plt.imshow(overlay)
plt.title("Overlay: Red = GT only, Green = Pred only, Yellow = Both")
plt.axis('off')
plt.tight_layout()
plt.savefig("C:/FAU/Project/mask_overlay_filtered.png", dpi=300)
plt.show()
