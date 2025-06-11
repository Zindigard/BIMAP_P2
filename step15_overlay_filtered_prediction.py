
import os
import numpy as np
from aicsimageio import AICSImage
from cellpose import models
from skimage.measure import find_contours
from read_roi import read_roi_file
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def polygon_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# Step 1: Calculate GT area range
roi_folder = "C:/FAU/Project/data sources/manually labeled/RoiSet_Contour_bacteria_ROI1"
roi_files = sorted([f for f in os.listdir(roi_folder) if f.endswith('.roi')])
gt_areas = []
gt_polygons = []

for filename in roi_files:
    roi_data = read_roi_file(os.path.join(roi_folder, filename))
    roi = list(roi_data.values())[0]
    x = np.array(roi['x'])
    y = np.array(roi['y'])
    if len(x) < 3:
        continue
    area = polygon_area(x, y)
    gt_areas.append(area)
    gt_polygons.append(np.stack([x, y], axis=1))

min_area = np.min(gt_areas)
max_area = np.max(gt_areas)

# Step 2: Segment image
czi_path = "C:/FAU/Project/data sources/ROI1/WT_NADA_RADA_HADA_NHS_40min_ROI1_SIM.czi"
img = AICSImage(czi_path)
rgb_image = np.stack([
    img.get_image_data("YX", S=0, T=0, C=0),
    img.get_image_data("YX", S=0, T=0, C=1),
    img.get_image_data("YX", S=0, T=0, C=2),
], axis=-1)

model = models.CellposeModel(gpu=False)
pred = model.eval(rgb_image, diameter=None)
masks = pred[0]

# Step 3: Filter predicted contours by area
contours = find_contours(masks, level=0.5)
filtered_contours = []

for cnt in contours:
    if len(cnt) < 5:
        continue
    cnt = np.fliplr(cnt)
    area = polygon_area(cnt[:, 0], cnt[:, 1])
    if min_area <= area <= max_area:
        filtered_contours.append(cnt)

# Step 4: Visualization
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(masks, cmap='gray')

# Draw predicted masks (filtered)
for cnt in filtered_contours:
    patch = Polygon(cnt, closed=True, edgecolor='lime', facecolor='none', linewidth=1.2)
    ax.add_patch(patch)

# Draw GT masks (in red)
for poly in gt_polygons:
    patch = Polygon(poly, closed=True, edgecolor='red', facecolor='none', linewidth=1.2, linestyle='--')
    ax.add_patch(patch)

ax.set_title("Overlay: Red = Ground Truth | Green = Predicted (Filtered by Area)")
plt.tight_layout()
plt.savefig("C:/FAU/Project/overlay_gt_vs_filtered_pred.png", dpi=300)
plt.show()
