import numpy as np
from czifile import imread
from cellpose import models
from skimage.measure import regionprops, label
from skimage.morphology import dilation, disk
from vedo import Ellipsoid, Plotter, settings

# --------- SETTINGS ---------
czi_path = "WT_NADA_RADA_HADA_NHS_40min_ROI1_SIM.czi"
depth = 30  # Height of ellipsoids in Z direction
min_cell_area = 20
# ----------------------------

# --- Load and prepare image ---
print(f"📂 Opening image: {czi_path}")
raw = imread(czi_path)
img = np.squeeze(raw)[0:3, :, :]  # (3, H, W)
print(f"✅ Final image shape: {img.shape}")

# Normalize and reorder channels (BGR to RGB)
def normalize(ch): return (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
rgb = np.stack([normalize(img[2]), normalize(img[1]), normalize(img[0])], axis=-1)
print(f"🖼️ RGB image shape: {rgb.shape}")

# --- Segment with Cellpose ---
print("🧠 Segmenting with Cellpose...")
model = models.Cellpose(gpu=False, model_type='cyto')
masks, _, _, _ = model.eval(
    rgb,
    diameter=None,
    flow_threshold=0.4,
    cellprob_threshold=0.5,
    channels=[0, 0]
)
print(f"✅ Segmentation complete. Detected {masks.max()} cells.")

# --- Morphological processing ---
masks = dilation(masks, disk(1))  # Fill gaps
labeled = label(masks)
props = regionprops(labeled)

# --- Create 3D ellipsoids ---
print("📐 Fitting ellipsoids to cells...")
actors = []
for prop in props:
    if prop.area < min_cell_area:
        continue

    y, x = prop.centroid
    ry = prop.major_axis_length / 2
    rx = prop.minor_axis_length / 2
    rz = (rx + ry) / 2.5  # Estimate depth

    region_mask = labeled == prop.label
    import random
    avg_color = [random.uniform(0.3, 1.0) for _ in range(3)]  # brighter random RGB

    ellip = Ellipsoid(
        pos=(x, y, depth // 2),
        axis1=(rx, 0, 0),
        axis2=(0, ry, 0),
        axis3=(0, 0, rz),
        c=avg_color,
        alpha=1
    ).lighting("plastic")
    actors.append(ellip)

# --- Configure lighting and display ---
print("🧼 Rendering 3D filled ellipsoids...")

settings.use_depth_peeling = True  # KEEP this

plt = Plotter(bg='white', axes=1)
plt.show(actors, "🦠 Filled Ellipsoid-Shaped Bacteria Cells", viewup='z')

