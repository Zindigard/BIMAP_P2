
import os
import numpy as np
from aicsimageio import AICSImage
from cellpose import models
from read_roi import read_roi_file
from skimage.draw import polygon as sk_polygon
from skimage.measure import regionprops, label
from vedo import Plotter, Mesh, Text2D
from sklearn.decomposition import PCA

def polygon_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def make_3d_sweep_around_axis(contour, num_steps=40, color="r"):
    if len(contour) < 3:
        return None
    contour = np.array(contour)
    pca = PCA(n_components=2)
    pca.fit(contour)
    pc1 = pca.components_[0]
    axis = np.array([pc1[0], pc1[1], 0])
    axis = axis / np.linalg.norm(axis)
    center = np.mean(contour, axis=0)
    contour_3d = np.column_stack([contour, np.zeros(len(contour))])

    angles = np.linspace(0, 2 * np.pi, num_steps)
    all_points = []
    for theta in angles:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        ux, uy, uz = axis
        rot = np.array([
            [cos_t + ux**2 * (1 - cos_t), ux*uy*(1 - cos_t), uy*sin_t],
            [ux*uy*(1 - cos_t), cos_t + uy**2*(1 - cos_t), -ux*sin_t],
            [-uy*sin_t, ux*sin_t, cos_t]
        ])
        rotated = (contour_3d - np.append(center, 0)) @ rot.T + np.append(center, 0)
        all_points.append(rotated)

    verts = np.concatenate(all_points, axis=0)
    n = len(contour)
    faces = []
    for i in range(num_steps - 1):
        for j in range(n):
            p1 = i * n + j
            p2 = i * n + (j + 1) % n
            p3 = (i + 1) * n + (j + 1) % n
            p4 = (i + 1) * n + j
            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])
    mesh = Mesh([verts, faces], c=color, alpha=0.6)
    mesh = mesh.subdivide(2)  # Subdivision smoothing
    return mesh

roi_folder = "C:/FAU/Project/data sources/manually labeled/RoiSet_Contour_bacteria_ROI1"
roi_files = sorted([f for f in os.listdir(roi_folder) if f.endswith('.roi')])

czi_path = "C:/FAU/Project/data sources/ROI1/WT_NADA_RADA_HADA_NHS_40min_ROI1_SIM.czi"
img = AICSImage(czi_path)
img_shape = img.get_image_data("YX", S=0, T=0, C=0).shape

gt_mask = np.zeros(img_shape, dtype=np.uint8)
gt_contours = []
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
    gt_contours.append(np.stack([x, y], axis=-1))
    gt_areas.append(polygon_area(x, y))

min_area = np.min(gt_areas)
max_area = np.max(gt_areas)

rgb_image = np.stack([
    img.get_image_data("YX", S=0, T=0, C=0),
    img.get_image_data("YX", S=0, T=0, C=1),
    img.get_image_data("YX", S=0, T=0, C=2),
], axis=-1)

model = models.CellposeModel(gpu=False)
pred = model.eval(rgb_image, diameter=None)
masks = pred[0]

filtered_pred_mask = np.zeros_like(masks, dtype=np.uint8)
labeled = label(masks)
regions = regionprops(labeled)
pred_contours = []

for r in regions:
    if min_area <= r.area <= max_area:
        coords = r.coords
        filtered_pred_mask[coords[:, 0], coords[:, 1]] = 1
        x = coords[:, 1]
        y = coords[:, 0]
        pred_contours.append(np.stack([x, y], axis=-1))

vp = Plotter(title="Subdivided 3D GT vs Filtered Predicted", axes=1, bg="white")
gt_count = 0
pred_count = 0

for cnt in gt_contours:
    mesh = make_3d_sweep_around_axis(cnt, color="blue")
    if mesh:
        vp += mesh
        gt_count += 1

for cnt in pred_contours:
    mesh = make_3d_sweep_around_axis(cnt, color="red")
    if mesh:
        vp += mesh
        pred_count += 1

vp += Text2D(f"GT: {gt_count} | Predicted: {pred_count}", pos="top-left", c="black", bg="lightgray")
vp.show()
