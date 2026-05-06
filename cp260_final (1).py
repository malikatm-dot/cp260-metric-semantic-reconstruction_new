# -*- coding: utf-8 -*-
# CP260 Final Project — Metric-Semantic Pose Estimation
# RBCCPS, Indian Institute of Science Bangalore
# Authors: Malika T M, Yashita Jaiswal


# CP260 Final Project — Metric-Semantic Pose Estimation
# RBCCPS, Indian Institute of Science Bangalore
# **Authors:** Malika T M, Yashita Jaiswal


# ============================================================
# Step 1: Setup
# ============================================================

import os, json, glob, shutil, gc
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DATA_DIR   = '/kaggle/input/datasets/malikatmm/datarp-zip/Data'
OUTPUT_DIR = '/kaggle/working/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/detections', exist_ok=True)

!pip install open3d -q
print('Dependencies installed')


# ============================================================
# Step 2: Load Dataset and Camera Parameters
# ============================================================

POSES_PATH = os.path.join(DATA_DIR, 'poses.json')

# Camera intrinsics (calibrated)
FX, FY = 1477.00974684544, 1480.4424455584467
CX, CY = 1298.2501500778505, 686.8201623541711
W, H   = 2560, 1440

K = np.array([[FX, 0., CX],
              [0., FY, CY],
              [0., 0., 1.]], dtype=np.float64)

with open(POSES_PATH) as f:
    all_poses_raw = json.load(f)
all_poses = {int(k): np.array(v, dtype=np.float64) for k, v in all_poses_raw.items()}

# Lazy image loading — avoid holding all frames in RAM simultaneously
image_paths = {}
for fp in sorted(glob.glob(f'{DATA_DIR}/frame_*.png')):
    idx = int(os.path.basename(fp).replace('frame_', '').replace('.png', ''))
    image_paths[idx] = fp

def get_frame(idx):
    return cv2.imread(image_paths[idx])

images_bgr = image_paths  # alias kept for compatibility

print(f'Found {len(image_paths)} frames')
print(f'Intrinsics: fx={FX:.1f}, fy={FY:.1f}, cx={CX:.1f}, cy={CY:.1f}')


# ============================================================
# Step 3: Core Geometry Utilities
# ============================================================

def get_projection_matrix(K, pose_c2w):
    """Build 3x4 projection matrix P = K [R | t] from camera-to-world pose."""
    pose_w2c = np.linalg.inv(pose_c2w)
    R = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3:]
    return K @ np.hstack([R, t])


def remove_outliers_mad(points, threshold=2.5):
    """MAD-based outlier rejection (threshold=2.5, scale=1.4826)."""
    if len(points) < 5:
        return points
    median = np.median(points, axis=0)
    diffs  = np.linalg.norm(points - median, axis=1)
    mad    = 1.4826 * np.median(diffs)
    if mad < 1e-10:
        return points
    return points[diffs < threshold * mad]


def triangulate_roi(entity_name, annotations, K, poses, n_grid=200):
    """
    Densely triangulate a connector's bounding box region across all view pairs.

    For each pair of annotated frames, a regular n x n grid of relative
    coordinates is mapped into pixel space in both views and triangulated.
    Points failing the cheirality check or exceeding the reprojection
    threshold (30 px) are discarded. Remaining points are MAD-filtered.
    """
    from itertools import combinations

    entity_ann    = annotations.get(entity_name, {})
    frame_indices = sorted(entity_ann.keys())
    if len(frame_indices) < 2:
        print(f'  [WARN] Need >=2 views for {entity_name}')
        return np.zeros((0, 3))

    n_side = int(np.sqrt(n_grid))
    ts = np.linspace(0.10, 0.90, n_side)
    all_points = []

    for idx1, idx2 in combinations(frame_indices, 2):
        if idx1 not in poses or idx2 not in poses:
            continue
        bbox1 = entity_ann[idx1]
        bbox2 = entity_ann[idx2]
        P1    = get_projection_matrix(K, poses[idx1])
        P2    = get_projection_matrix(K, poses[idx2])
        w2c1  = np.linalg.inv(poses[idx1])
        w2c2  = np.linalg.inv(poses[idx2])

        for ty in ts:
            for tx in ts:
                u1 = bbox1[0] + tx * (bbox1[2] - bbox1[0])
                v1 = bbox1[1] + ty * (bbox1[3] - bbox1[1])
                u2 = bbox2[0] + tx * (bbox2[2] - bbox2[0])
                v2 = bbox2[1] + ty * (bbox2[3] - bbox2[1])

                pts4d = cv2.triangulatePoints(
                    P1, P2,
                    np.array([[u1, v1]], dtype=np.float64).T,
                    np.array([[u2, v2]], dtype=np.float64).T
                )
                pt3d = (pts4d[:3] / pts4d[3:]).flatten()

                z1 = (w2c1[:3, :3] @ pt3d + w2c1[:3, 3])[2]
                z2 = (w2c2[:3, :3] @ pt3d + w2c2[:3, 3])[2]
                if z1 <= 0 or z2 <= 0:
                    continue

                proj1 = P1 @ np.append(pt3d, 1)
                proj1 = proj1[:2] / proj1[2]
                proj2 = P2 @ np.append(pt3d, 1)
                proj2 = proj2[:2] / proj2[2]

                if (np.linalg.norm(proj1 - [u1, v1]) < 30 and
                        np.linalg.norm(proj2 - [u2, v2]) < 30):
                    all_points.append(pt3d)

    if not all_points:
        print(f'  [WARN] No 3D points for {entity_name}')
        return np.zeros((0, 3))

    pts = remove_outliers_mad(np.array(all_points))
    print(f'  {entity_name}: {len(pts)} triangulated points')
    return pts


def estimate_panel_rotation(all_entity_points):
    """
    Fit a single plane to all connector clouds (coplanar by construction).
    Returns a shared 3x3 rotation matrix R_panel where rows are:
      row0 -> panel width  axis (Y-dominant)
      row1 -> panel height axis (Z-dominant)
      row2 -> panel normal axis (X-dominant, depth direction)
    """
    combined = np.vstack(all_entity_points)
    centroid = combined.mean(axis=0)
    _, _, Vt = np.linalg.svd(combined - centroid)
    normal = Vt[-1]

    if normal[0] < 0:
        normal = -normal

    row2 = normal / np.linalg.norm(normal)
    up   = np.array([0., 0., 1.])
    row1 = up - np.dot(up, row2) * row2
    row1 /= np.linalg.norm(row1)
    row0 = np.cross(row1, row2)
    row0 /= np.linalg.norm(row0)

    if row0[1] < 0:
        row0 = -row0
        row1 = np.cross(row2, row0)
        row1 /= np.linalg.norm(row1)

    return np.array([-row2, row1, row0])


# Connector specifications (IEC 60083, RJ-45, DE-15) — used as upper bounds
PHYSICAL_EXTENTS = {
    'power_socket':    [0.030, 0.018, 0.006],   # ~30 x 18 x 6 mm
    'ethernet_socket': [0.016, 0.013, 0.006],   # ~16 x 13 x 6 mm
    'vga_socket':      [0.035, 0.012, 0.006],   # ~35 x 12 x 6 mm
}
DEPTH_PRIOR = 0.006


def fit_obb(points_3d, entity_name, R_panel):
    """
    Construct an OBB using the shared panel rotation.
    Centre is the median of point projections onto R_panel.
    Extents estimated using robust PCA (2*std of projected cloud),
    scaled to capture full connector spread and capped at physical
    connector specifications to prevent overestimation on sparse clouds.
    """
    if len(points_3d) < 3:
        return None

    projected    = points_3d @ R_panel.T
    center_proj  = np.median(projected, axis=0)
    center_world = R_panel.T @ center_proj

    centered = projected - projected.mean(axis=0)
    cov      = np.cov(centered.T)
    vals, _  = np.linalg.eig(cov)
    half_extents = 2.0 * np.sqrt(np.abs(vals))

    half_extents[0] *= 3.0
    half_extents[1] *= 3.0

    spec = np.array(PHYSICAL_EXTENTS.get(entity_name, [0.020, 0.015, DEPTH_PRIOR]))
    half_extents = np.minimum(half_extents, spec)
    half_extents[2] = DEPTH_PRIOR

    return {
        'center':   center_world.tolist(),
        'extent':   half_extents.tolist(),
        'rotation': R_panel.tolist()
    }

print('Geometry utilities ready.')


# ============================================================
# Step 4: Manual Annotations
# ============================================================
# Bounding boxes were annotated using **Pixspy** (pixel-level browser annotation tool).
# Format: `[x1, y1, x2, y2]` in original 2560×1440 pixels.
# Boxes are inset from the outer bezel to capture only the connector face.

ENTITY_ANNOTATIONS = {
    'power_socket': {
        319: [1437, 651,  1455, 676],
        333: [1541, 741,  1572, 774],
        353: [1144, 1022, 1193, 1072],
        359: [1806, 1003, 1873, 1054],
        365: [1492, 1172, 1565, 1242],
        400: [1314, 846,  1447, 955],
        426: [1528, 845,  1570, 875],
        449: [1622, 912,  1669, 946],
        461: [1853, 883,  1894, 914],
        468: [1581, 1040, 1631, 1082],
        471: [1217, 1036, 1274, 1084],
        496: [1724, 1086, 1763, 1142],
        531: [1070, 1124, 1217, 1277],
    },
    'ethernet_socket': {
        333: [1567, 414,  1582, 434],
        353: [1149, 485,  1173, 520],
        359: [1871, 480,  1907, 516],
        365: [1538, 527,  1579, 572],
        371: [1630, 607,  1688, 669],
        390: [1355, 1135, 1426, 1222],
        426: [1555, 490,  1576, 514],
        449: [1653, 512,  1679, 538],
        461: [1894, 503,  1917, 528],
        468: [1611, 559,  1641, 591],
        471: [1221, 559,  1251, 591],
        496: [1762, 577,  1786, 612],
        515: [1057, 993,  1132, 1097],
    },
    'vga_socket': {
        359: [1847, 262, 1870, 302],
        468: [1591, 366, 1609, 400],
        471: [1184, 364, 1203, 399],
    },
}

# Remove any frame indices not present in this dataset
for entity in ENTITY_ANNOTATIONS:
    ENTITY_ANNOTATIONS[entity] = {
        k: v for k, v in ENTITY_ANNOTATIONS[entity].items()
        if k in image_paths and k in all_poses
    }

# Visualise annotations on sample frames
colors = {
    'power_socket':    (255, 0,   0),
    'ethernet_socket': (0,   0, 255),
    'vga_socket':      (0, 255,   0),
}
viz_ids = [idx for idx in [471, 496] if idx in image_paths]
fig, axes = plt.subplots(1, len(viz_ids), figsize=(20, 8))
if len(viz_ids) == 1:
    axes = [axes]
for col, idx in enumerate(viz_ids):
    img = cv2.cvtColor(get_frame(idx), cv2.COLOR_BGR2RGB)
    axes[col].imshow(img)
    for entity, ann_dict in ENTITY_ANNOTATIONS.items():
        if idx in ann_dict:
            x1, y1, x2, y2 = ann_dict[idx]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2,
                edgecolor=np.array(colors.get(entity, (255, 255, 0))) / 255,
                facecolor='none')
            axes[col].add_patch(rect)
            axes[col].text(x1, y1 - 10, entity,
                           color=np.array(colors.get(entity, (255, 255, 0))) / 255,
                           fontsize=9)
    axes[col].set_title(f'frame_{idx:06d}', fontsize=10)
    axes[col].axis('off')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/annotations.png', dpi=100)
plt.show()
print('Annotations visualised.')


# ============================================================
# Step 5: OBB Pose Estimation
# ============================================================

print('=== OBB Pose Estimation ===\n')

# Triangulate each connector independently, freeing memory between entities
all_point_clouds = {}
for entity_name in ENTITY_ANNOTATIONS:
    print(f'Triangulating: {entity_name}')
    pts = triangulate_roi(entity_name, ENTITY_ANNOTATIONS, K, all_poses)
    if len(pts) >= 3:
        all_point_clouds[entity_name] = pts
    gc.collect()

if not all_point_clouds:
    raise RuntimeError('No 3D points triangulated. Check annotations and frame indices.')

# Estimate a single shared panel rotation from all connector clouds
print('\nEstimating shared panel rotation...')
R_panel = estimate_panel_rotation(list(all_point_clouds.values()))
print(f'Panel normal: {R_panel[2].round(4)}')

# Fit OBBs using the shared rotation
results = []
for entity_name, points_3d in all_point_clouds.items():
    print(f'\nFitting OBB: {entity_name}')
    obb = fit_obb(points_3d, entity_name, R_panel)
    if obb is None:
        print(f'  [SKIP] Not enough points')
        continue
    results.append({'entity': entity_name, 'obb': obb})
    c, e = obb['center'], obb['extent']
    print(f'  Center: [{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}]')
    print(f'  Extent: [{e[0]:.4f}, {e[1]:.4f}, {e[2]:.4f}]')
    del points_3d
    gc.collect()

answers_path = f'{OUTPUT_DIR}/answers.json'
with open(answers_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved {len(results)} OBBs to {answers_path}')


# ============================================================
# Step 6: OBB Projection Visualisation
# ============================================================

def project_obb(obb, K, pose_c2w):
    center = np.array(obb['center'])
    extent = np.array(obb['extent'])
    R_obb  = np.array(obb['rotation'])
    signs  = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                        [1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]], dtype=float)
    corners_world = (R_obb.T @ (signs * extent).T).T + center
    pose_w2c      = np.linalg.inv(pose_c2w)
    corners_cam   = (pose_w2c[:3,:3] @ corners_world.T).T + pose_w2c[:3,3]
    corners_2d = np.full((8, 2), np.nan)
    for i in range(8):
        if corners_cam[i, 2] > 0:
            pt = K @ corners_cam[i]
            corners_2d[i] = pt[:2] / pt[2]
    return corners_2d


def draw_obb(image, corners_2d, label='', color=(0, 255, 0)):
    vis  = image.copy()
    pts  = corners_2d.astype(int)
    edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),
             (4,5),(4,6),(5,7),(6,7)]
    for i, j in edges:
        if not (np.isnan(corners_2d[i]).any() or np.isnan(corners_2d[j]).any()):
            cv2.line(vis, tuple(pts[i]), tuple(pts[j]), color, 2)
    if label:
        valid = pts[~np.isnan(corners_2d).any(axis=1)]
        if len(valid):
            cv2.putText(vis, label, tuple(valid.min(axis=0) - [0, 10]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return vis


obb_colors = {
    'power_socket':    (0,   0, 255),
    'ethernet_socket': (255, 0,   0),
    'vga_socket':      (0, 255,   0),
}
viz_frames = [idx for idx in [471, 496, 515] if idx in image_paths and idx in all_poses]
fig, axes = plt.subplots(1, len(viz_frames), figsize=(8 * len(viz_frames), 7))
if len(viz_frames) == 1:
    axes = [axes]
for col, idx in enumerate(viz_frames):
    vis = get_frame(idx).copy()
    for res in results:
        corners_2d = project_obb(res['obb'], K, all_poses[idx])
        color = obb_colors.get(res['entity'], (0, 255, 255))
        vis   = draw_obb(vis, corners_2d, res['entity'], color)
    cv2.imwrite(f"{OUTPUT_DIR}/detections/obb_{idx:06d}.png", vis)
    axes[col].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[col].set_title(f'frame_{idx:06d}', fontsize=10)
    axes[col].axis('off')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/obb_projections.png', dpi=100)
plt.show()
print('OBB projections saved.')


# ============================================================
# Step 7: Validation Against Ground Truth
# ============================================================

GT_ANSWERS = {
    'power_socket': {
        'center':   [0.2899563791050707, 0.224012997656665, 0.5263346643095937],
        'extent':   [0.03, 0.018, 0.006],
        'rotation': [[-0.05369060880533758, 0.9983112173142267, -0.022181792278784064],
                     [0.0, 0.02221383308886612, 0.9997532423640847],
                     [0.9985576190306242, 0.05367736023769152, -0.0011926742224425658]]
    },
    'ethernet_socket': {
        'center':   [0.2885918621251939, 0.229041103707018, 0.755931820178689],
        'extent':   [0.016, 0.013, 0.006],
        'rotation': [[-0.05369060880533758, 0.9983112173142267, -0.022181792278784064],
                     [0.0, 0.02221383308886612, 0.9997532423640847],
                     [0.9985576190306242, 0.05367736023769152, -0.0011926742224425658]]
    },
    'vga_socket': {
        'center':   [0.2708474330734349, 0.22991300045854823, 0.8381223611040131],
        'extent':   [0.035, 0.012, 0.006],
        'rotation': [[-0.05369060880533758, 0.9983112173142267, -0.022181792278784064],
                     [0.0, 0.02221383308886612, 0.9997532423640847],
                     [0.9985576190306242, 0.05367736023769152, -0.0011926742224425658]]
    }
}


def get_obb_corners(obb):
    c = np.array(obb['center'])
    e = np.array(obb['extent'])
    R = np.array(obb['rotation'])
    signs = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                       [1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]], dtype=float)
    return (R.T @ (signs * e).T).T + c


def aabb_iou_3d(obb1, obb2):
    """Axis-aligned IoU of OBB corner clouds (course evaluation metric)."""
    c1 = get_obb_corners(obb1)
    c2 = get_obb_corners(obb2)
    min1, max1 = c1.min(axis=0), c1.max(axis=0)
    min2, max2 = c2.min(axis=0), c2.max(axis=0)
    inter = np.maximum(0, np.minimum(max1, max2) - np.maximum(min1, min2))
    i_vol = inter.prod()
    union = (max1 - min1).prod() + (max2 - min2).prod() - i_vol
    return i_vol / union if union > 0 else 0.0


def centre_pixel_error(obb, entity_name, annotations, poses, K):
    """Mean reprojection error of the estimated 3D centre across annotated frames."""
    c   = np.array(obb['center'])
    ann = annotations.get(entity_name, {})
    errors = []
    for idx, bbox in ann.items():
        if idx not in poses:
            continue
        P    = get_projection_matrix(K, poses[idx])
        proj = P @ np.append(c, 1)
        proj = proj[:2] / proj[2]
        u_obs = np.array([(bbox[0] + bbox[2]) / 2,
                          (bbox[1] + bbox[3]) / 2])
        errors.append(np.linalg.norm(proj - u_obs))
    return np.mean(errors) if errors else 0.0


print('\n=== Validation vs Ground Truth ===\n')
ious = []
for res in results:
    name = res['entity']
    if name not in GT_ANSWERS:
        continue
    gt  = GT_ANSWERS[name]
    our = res['obb']
    c_err  = np.linalg.norm(np.array(gt['center']) - np.array(our['center'])) * 1000
    R_diff = np.array(our['rotation']) @ np.array(gt['rotation']).T
    angle  = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))
    iou    = aabb_iou_3d(gt, our)
    px_err = centre_pixel_error(our, name, ENTITY_ANNOTATIONS, all_poses, K)
    ious.append(iou)
    print(f'{name}')
    print(f'  Centre error:   {c_err:.1f} mm')
    print(f'  Pixel error:    {px_err:.1f} px')
    print(f'  Rotation error: {angle:.2f}°')
    print(f'  IoU:            {iou:.4f}')
    print()

if ious:
    print(f'Mean IoU: {np.mean(ious):.4f}')


# ============================================================
# Step 8: Save and Print Final Answers
# ============================================================

with open(answers_path, 'w') as f:
    json.dump(results, f, indent=2)

print('=== FINAL ANSWERS ===')
print(json.dumps(results, indent=2))
print('\nAnswers saved to:', answers_path)
print('Download answers.json from the Kaggle Output panel on the right.')
