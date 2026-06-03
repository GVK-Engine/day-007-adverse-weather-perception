# weather_benchmark.py
# Measures how rain and fog degrade LiDAR detection accuracy
# Runs my Day 1 detector on KITTI across weather conditions
# Looking for the exact intensity where detection becomes unsafe
#
# Vamshikrishna Gadde | Day 7 of 90 | MS Robotics ASU

import numpy as np
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rain_simulator import load_kitti_lidar, simulate_rain, simulate_fog

KITTI_DIR = (
    r"C:\Users\vamsh\Downloads\kitti"
    r"\2011_09_26_drive_0001_sync"
    r"\2011_09_26"
    r"\2011_09_26_drive_0001_sync"
    r"\velodyne_points\data"
)

RESULTS_DIR      = "results"
VOXEL_SIZE       = 0.2
RANSAC_DIST      = 0.3
DBSCAN_EPS       = 0.5
DBSCAN_MIN_PTS   = 10
SAFETY_THRESHOLD = 0.85  # below 85% of baseline = unsafe


def voxel_downsample(points, voxel_size):
    # keep one point per voxel grid cell
    coords = np.floor(points[:, :3] / voxel_size).astype(int)
    _, idx = np.unique(coords, axis=0, return_index=True)
    return points[idx]


def remove_ground_ransac(points, dist_thresh=0.3, n_iter=100):
    # RANSAC to find and remove the ground plane
    # samples 3 random points, fits plane, keeps best
    best_mask  = np.zeros(len(points), dtype=bool)
    best_count = 0

    for _ in range(n_iter):
        idx     = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[idx, :3]
        v1, v2  = p2 - p1, p3 - p1
        n       = np.cross(v1, v2)

        if np.linalg.norm(n) < 1e-6:
            continue

        n     = n / np.linalg.norm(n)
        dists = np.abs(points[:, :3] @ n - np.dot(n, p1))
        mask  = dists < dist_thresh

        if mask.sum() > best_count:
            best_count = mask.sum()
            best_mask  = mask

    return points[~best_mask]


def dbscan_cluster(points, eps=0.5, min_pts=10):
    # basic DBSCAN — slow but no dependencies
    # NOTE: scipy version would be 100x faster
    #       keeping this pure numpy for transparency
    if len(points) == 0:
        return []

    n        = len(points)
    labels   = -np.ones(n, dtype=int)
    visited  = np.zeros(n, dtype=bool)
    cid      = 0

    def get_neighbors(i):
        d = np.sqrt(((points[:, :3] - points[i, :3])**2).sum(axis=1))
        return np.where(d < eps)[0]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nbrs = get_neighbors(i)

        if len(nbrs) < min_pts:
            continue

        labels[i] = cid
        queue = list(nbrs)

        while queue:
            j = queue.pop(0)
            if not visited[j]:
                visited[j] = True
                nbrs_j = get_neighbors(j)
                if len(nbrs_j) >= min_pts:
                    queue.extend(nbrs_j.tolist())
            if labels[j] == -1:
                labels[j] = cid

        cid += 1

    return [points[labels == c] for c in range(cid)]


def detect_objects(points):
    # full pipeline: downsample → remove ground → cluster
    # returns valid clusters and runtime
    t0 = time.time()

    if len(points) < 100:
        return [], time.time() - t0

    ds    = voxel_downsample(points, VOXEL_SIZE)
    above = remove_ground_ransac(ds, RANSAC_DIST)

    # height filter — above ground, below rooftop
    above = above[(above[:, 2] > -1.5) & (above[:, 2] < 3.0)]

    if len(above) < 50:
        return [], time.time() - t0

    clusters = dbscan_cluster(above, DBSCAN_EPS, DBSCAN_MIN_PTS)

    # filter by cluster size — ignore noise and walls
    valid = [c for c in clusters if 20 < len(c) < 2000]

    return valid, time.time() - t0


def run_benchmark(frame_paths):
    # test each frame at all rain and fog levels
    # compare detection count vs clean baseline

    rain_levels = [0, 10, 25, 50, 75, 100]
    fog_levels  = [200, 150, 100, 75, 50, 25, 10]

    rain_rates = {r: [] for r in rain_levels}
    fog_rates  = {v: [] for v in fog_levels}

    print(f"\n  Testing {len(frame_paths)} frames × "
          f"{len(rain_levels)} rain + {len(fog_levels)} fog levels")
    print(f"  {'─'*55}")

    for fi, fpath in enumerate(frame_paths):
        pts      = load_kitti_lidar(fpath)
        fname    = os.path.basename(fpath)
        baseline, _ = detect_objects(pts)
        n_base   = len(baseline)

        if n_base == 0:
            print(f"  Frame {fi+1:02d} — no objects detected, skipping")
            continue

        print(f"\n  Frame {fi+1:02d}/{len(frame_paths)} "
              f"({fname[:14]}) baseline={n_base} objects")

        for r in rain_levels:
            deg, _ = simulate_rain(pts, r, seed=fi)
            det, _ = detect_objects(deg)
            rate   = min(len(det) / n_base, 1.0)
            rain_rates[r].append(rate)

        for v in fog_levels:
            deg, _ = simulate_fog(pts, v, seed=fi)
            det, _ = detect_objects(deg)
            rate   = min(len(det) / n_base, 1.0)
            fog_rates[v].append(rate)

        # quick summary for this frame
        summary = "  ".join([
            f"{r}:{rain_rates[r][-1]*100:.0f}%"
            for r in rain_levels
        ])
        print(f"    rain → {summary}")

    rain_mean = {r: np.mean(v) for r, v in rain_rates.items() if v}
    fog_mean  = {v: np.mean(r) for v, r in fog_rates.items()  if r}

    return rain_mean, fog_mean


def find_unsafe_threshold(results, threshold=SAFETY_THRESHOLD):
    # returns first intensity where detection drops below threshold
    for intensity, rate in sorted(results.items()):
        if rate < threshold:
            return intensity
    return None


def plot_results(rain_mean, fog_mean):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#1a1a1a')
    fig.suptitle(
        "Adverse Weather LiDAR Perception Analysis — KITTI\n"
        "Vamshikrishna Gadde  |  MS Robotics ASU  |  Day 7 of 90",
        fontsize=13, color='white'
    )

    def style_ax(ax):
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')

    # rain chart
    ax1   = axes[0]
    rx    = sorted(rain_mean.keys())
    ry    = [rain_mean[r] * 100 for r in rx]
    style_ax(ax1)

    ax1.plot(rx, ry, 'o-', color='#00C8FF', linewidth=2.5,
             markersize=8, markerfacecolor='white',
             markeredgecolor='#00C8FF', markeredgewidth=2)
    ax1.axhline(y=SAFETY_THRESHOLD * 100, color='red',
                linestyle='--', linewidth=2,
                label=f'Safety limit ({SAFETY_THRESHOLD*100:.0f}%)')
    ax1.fill_between(rx,
                     [SAFETY_THRESHOLD * 100] * len(rx),
                     [max(0, min(ry) - 5)] * len(rx),
                     alpha=0.15, color='red', label='Unsafe zone')

    thresh = find_unsafe_threshold(rain_mean)
    if thresh:
        ax1.axvline(x=thresh, color='orange', linestyle=':',
                    linewidth=2, label=f'Breaks at {thresh} mm/hr')

    for x, y in zip(rx, ry):
        ax1.annotate(f'{y:.0f}%', xy=(x, y),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', color='white', fontsize=9)

    ax1.set_xlabel('Rain Intensity (mm/hr)', color='white')
    ax1.set_ylabel('Detection Rate vs Clear Baseline (%)',
                   color='white')
    ax1.set_title('Rain Impact on LiDAR Detection',
                  color='white', fontsize=12)
    ax1.set_ylim(0, 115)
    ax1.legend(facecolor='#1a1a1a', labelcolor='white',
               fontsize=9, edgecolor='#444')

    # fog chart
    ax2 = axes[1]
    fx  = sorted(fog_mean.keys(), reverse=True)
    fy  = [fog_mean[v] * 100 for v in fx]
    style_ax(ax2)

    ax2.plot(fx, fy, 's-', color='#FF6B35', linewidth=2.5,
             markersize=8, markerfacecolor='white',
             markeredgecolor='#FF6B35', markeredgewidth=2)
    ax2.axhline(y=SAFETY_THRESHOLD * 100, color='red',
                linestyle='--', linewidth=2,
                label=f'Safety limit ({SAFETY_THRESHOLD*100:.0f}%)')
    ax2.fill_between(fx,
                     [SAFETY_THRESHOLD * 100] * len(fx),
                     [max(0, min(fy) - 5)] * len(fy),
                     alpha=0.15, color='red', label='Unsafe zone')

    fog_thresh = find_unsafe_threshold(fog_mean)
    if fog_thresh:
        ax2.axvline(x=fog_thresh, color='orange', linestyle=':',
                    linewidth=2,
                    label=f'Breaks at {fog_thresh}m visibility')

    for x, y in zip(fx, fy):
        ax2.annotate(f'{y:.0f}%', xy=(x, y),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', color='white', fontsize=9)

    ax2.set_xlabel('Visibility (m)', color='white')
    ax2.set_ylabel('Detection Rate vs Clear Baseline (%)',
                   color='white')
    ax2.set_title('Fog Impact on LiDAR Detection',
                  color='white', fontsize=12)
    ax2.set_ylim(0, 115)
    ax2.legend(facecolor='#1a1a1a', labelcolor='white',
               fontsize=9, edgecolor='#444')

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, 'weather_benchmark.png')
    plt.savefig(path, dpi=130, bbox_inches='tight',
                facecolor='#1a1a1a')
    plt.close()
    print(f"\n  Saved: {path}")
    return path


if __name__ == "__main__":

    print("\n" + "="*58)
    print("  Adverse Weather Perception Benchmark")
    print("  Day 7 of 90")
    print("="*58)

    files = sorted([
        os.path.join(KITTI_DIR, f)
        for f in os.listdir(KITTI_DIR)
        if f.endswith('.bin')
    ])[:10]

    print(f"  Frames    : {len(files)}")
    print(f"  Threshold : {SAFETY_THRESHOLD*100:.0f}%")

    rain_mean, fog_mean = run_benchmark(files)

    print(f"\n  {'='*58}")
    print(f"  RAIN RESULTS")
    print(f"  {'mm/hr':<12} {'Rate':<14} {'Status'}")
    print(f"  {'─'*40}")
    for r in sorted(rain_mean):
        rate   = rain_mean[r]
        status = "✅ Safe" if rate >= SAFETY_THRESHOLD else "❌ UNSAFE"
        print(f"  {r:<12} {rate*100:<14.1f}% {status}")

    print(f"\n  FOG RESULTS")
    print(f"  {'Vis (m)':<12} {'Rate':<14} {'Status'}")
    print(f"  {'─'*40}")
    for v in sorted(fog_mean, reverse=True):
        rate   = fog_mean[v]
        status = "✅ Safe" if rate >= SAFETY_THRESHOLD else "❌ UNSAFE"
        print(f"  {v:<12} {rate*100:<14.1f}% {status}")

    rain_thresh = find_unsafe_threshold(rain_mean)
    fog_thresh  = find_unsafe_threshold(fog_mean)

    print(f"\n  KEY FINDINGS")
    print(f"  {'─'*40}")
    if rain_thresh:
        print(f"  Rain unsafe at : {rain_thresh} mm/hr")
    else:
        print(f"  Rain stays safe across all tested levels")
    if fog_thresh:
        print(f"  Fog unsafe at  : {fog_thresh}m visibility")
    else:
        print(f"  Fog stays safe across all tested levels")

    print(f"\n  This is the Operational Design Domain boundary.")
    print(f"  Waymo defines ODD limits exactly like this.")

    plot_results(rain_mean, fog_mean)
    print("="*58)