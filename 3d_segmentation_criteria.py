import numpy as np
from czifile import imread
from cellpose import models
from skimage.measure import regionprops, label
from skimage.morphology import dilation, disk
from vedo import Ellipsoid, Plotter, settings
import random

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
metrics = []
for prop in props:
    if prop.area < min_cell_area:
        continue

    y, x = prop.centroid
    ry = prop.major_axis_length / 2
    rx = prop.minor_axis_length / 2
    rz = (rx + ry) / 2.5  # Estimate depth

    # Region mask for color averaging
    region_mask = labeled == prop.label
    avg_color = [random.uniform(0.3, 1.0) for _ in range(3)]  # brighter random RGB

    # Create the ellipsoid
    ellip = Ellipsoid(
        pos=(x, y, depth // 2),
        axis1=(rx, 0, 0),
        axis2=(0, ry, 0),
        axis3=(0, 0, rz),
        c=avg_color,
        alpha=1
    ).lighting("plastic")
    
    actors.append(ellip)

    # --- Compute Shape Metrics (Volume, Sphericity, Eccentricity) ---
    volume = (4/3) * np.pi * rx * ry * rz  # Ellipsoid volume formula
    sphericity = (np.pi**(1/3) * (6 * volume)**(2/3)) / (4 * np.pi * (rx + ry))  # Simplified spherical surface area
    eccentricity = np.sqrt(1 - (min(rx, ry) / max(rx, ry))**2)  # Simplified eccentricity calculation

    metrics.append({
        'Cell': prop.label,
        'Volume': volume,
        'Sphericity': sphericity,
        'Eccentricity': eccentricity
    })

# --- Configure lighting and display ---
print("🧼 Rendering 3D filled ellipsoids...")

settings.use_depth_peeling = True  # KEEP this

plt = Plotter(bg='white', axes=1)
plt.show(actors, "🦠 Filled Ellipsoid-Shaped Bacteria Cells", viewup='z')

# --- Display Metrics ---
print("\n--- Shape Descriptors for Segmented Cells ---")
for metric in metrics:
    print(f"Cell {metric['Cell']}: Volume = {metric['Volume']:.2f}, Sphericity = {metric['Sphericity']:.2f}, Eccentricity = {metric['Eccentricity']:.2f}")

# === Averages of Predicted Metrics ===
avg_volume = np.mean([metric['Volume'] for metric in metrics])
avg_sphericity = np.mean([metric['Sphericity'] for metric in metrics])
avg_eccentricity = np.mean([metric['Eccentricity'] for metric in metrics])

# === Expected Metrics for Streptococcus pneumoniae ===
expected_volume = 0.18  # µm³ (Example)
expected_sphericity = 1  # Spherical shape
expected_eccentricity = 0  # Near spherical

# === Comparison of Predicted and Expected Metrics ===
print("\n--- Comparison of Predicted vs. Expected Metrics ---")
print(f"Average Predicted Volume: {avg_volume:.2f} µm³ (Expected: {expected_volume} µm³)")
print(f"Average Predicted Sphericity: {avg_sphericity:.2f} (Expected: {expected_sphericity})")
print(f"Average Predicted Eccentricity: {avg_eccentricity:.2f} (Expected: {expected_eccentricity})")
