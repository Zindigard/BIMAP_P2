
import os
from vedo import Plotter, Mesh, settings
import numpy as np
from read_roi import read_roi_file
from sklearn.decomposition import PCA

settings.default_font = "Courier"

def rotate_xy(p, center, angle_rad):
    p = np.array(p)
    c = np.array(center)
    v = p - c
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rot = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ])
    return rot @ v + c

def make_revolved_mesh(points_2d, resolution=60, n_angles=60, color='limegreen'):
    pca = PCA(n_components=2)
    pca.fit(points_2d)
    center = pca.mean_
    direction = pca.components_[0]
    angle_to_y = np.arctan2(direction[0], direction[1])

    aligned = []
    for p in points_2d:
        v = p - center
        cos_a = np.cos(-angle_to_y)
        sin_a = np.sin(-angle_to_y)
        rot = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])
        aligned.append(rot @ v)
    aligned = np.array(aligned)

    y_sorted_idx = np.argsort(aligned[:,1])
    r_profile = np.abs(aligned[y_sorted_idx, 0])
    z_profile = aligned[y_sorted_idx, 1]

    if len(z_profile) < 5:
        return None

    z_lin = np.linspace(z_profile.min(), z_profile.max(), resolution)
    r_interp = np.interp(z_lin, z_profile, r_profile)

    theta = np.linspace(0, 2*np.pi, n_angles)
    T, Z = np.meshgrid(theta, z_lin)
    R = np.tile(r_interp, (n_angles, 1)).T
    X = R * np.cos(T)
    Y = R * np.sin(T)

    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    faces = []
    for i in range(resolution - 1):
        for j in range(n_angles - 1):
            a = i * n_angles + j
            b = a + n_angles
            faces.append([a, b, b + 1])
            faces.append([a, b + 1, a + 1])

    points_rotated = []
    for p in points:
        xz_rotated = rotate_xy([p[0], p[2]], [0,0], angle_to_y)
        points_rotated.append([xz_rotated[0] + center[0], p[1] + center[1], xz_rotated[1]])

    return Mesh([points_rotated, faces], c=color, alpha=0.9)

roi_folder = "C:/FAU/Project/data sources/manually labeled/RoiSet_Contour_bacteria_ROI1"
roi_files = sorted([f for f in os.listdir(roi_folder) if f.endswith('.roi')])[:10]

plt = Plotter(title="Smooth Comparison: Laplacian vs Subdivision", axes=1, bg="white")

for idx, filename in enumerate(roi_files):
    roi_path = os.path.join(roi_folder, filename)
    roi_data = read_roi_file(roi_path)
    roi = list(roi_data.values())[0]
    x = np.array(roi['x'])
    y = np.array(roi['y'])
    if len(x) < 5:
        continue
    points = np.stack([x, y], axis=1)
    mesh = make_revolved_mesh(points)
    if mesh:
        if idx % 2 == 0:
            mesh.smooth(niter=30)
        else:
            mesh.subdivide(n=2)
        plt += mesh

plt.show(interactive=True)
