# visualize_weather.py
# Builds two outputs:
#   1. Safety heatmap — 2D operational design domain
#      shows which rain+fog combinations are safe to operate
#   2. Animated GIF — point cloud degrading through weather
#      visual proof of what weather does to LiDAR
#
# Nani | Day 7 of 90 | MS Robotics ASU

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio.v2 as imageio
from rain_simulator import load_kitti_lidar, simulate_rain, simulate_fog

KITTI_DIR = (
    r"C:\Users\vamsh\Downloads\kitti"
    r"\2011_09_26_drive_0001_sync"
    r"\2011_09_26"
    r"\2011_09_26_drive_0001_sync"
    r"\velodyne_points\data"
)
RESULTS_DIR      = "results"
SAFETY_THRESHOLD = 0.85
VOXEL_SIZE       = 0.2


def voxel_downsample(points, voxel_size):
    coords = np.floor(points[:, :3] / voxel_size).astype(int)
    _, idx = np.unique(coords, axis=0, return_index=True)
    return points[idx]


def remove_ground(points, dist_thresh=0.3, n_iter=80):
    best_mask  = np.zeros(len(points), dtype=bool)
    best_count = 0
    for _ in range(n_iter):
        idx = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[idx, :3]
        n = np.cross(p2 - p1, p3 - p1)
        if np.linalg.norm(n) < 1e-6:
            continue
        n     = n / np.linalg.norm(n)
        dists = np.abs(points[:, :3] @ n - np.dot(n, p1))
        mask  = dists < dist_thresh
        if mask.sum() > best_count:
            best_count = mask.sum()
            best_mask  = mask
    return points[~best_mask]


def simple_cluster_count(points):
    if len(points) < 50:
        return 0
    ds    = voxel_downsample(points, VOXEL_SIZE)
    above = remove_ground(ds)
    above = above[(above[:, 2] > -1.5) & (above[:, 2] < 3.0)]
    if len(above) < 30:
        return 0
    grid = {}
    for p in above:
        key = (int(p[0]), int(p[1]))
        grid[key] = grid.get(key, 0) + 1
    return sum(1 for v in grid.values() if v >= 3)


def build_safety_heatmap(frame_paths):
    rain_levels = [0, 10, 25, 50, 75, 100]
    fog_levels  = [200, 150, 100, 75, 50, 25, 10]
    scores      = np.zeros((len(fog_levels), len(rain_levels)))

    print(f"  Building safety heatmap...")
    print(f"  {len(rain_levels)} rain x {len(fog_levels)} fog "
          f"= {len(rain_levels)*len(fog_levels)} combinations")

    for fi, fpath in enumerate(frame_paths):
        pts      = load_kitti_lidar(fpath)
        baseline = simple_cluster_count(pts)
        if baseline == 0:
            continue
        print(f"  Frame {fi+1}/{len(frame_paths)} "
              f"baseline={baseline}")

        for ri, rain in enumerate(rain_levels):
            for fogi, fog in enumerate(fog_levels):
                deg = pts.copy()
                if rain > 0:
                    deg, _ = simulate_rain(deg, rain, seed=fi)
                if fog < 200:
                    deg, _ = simulate_fog(deg, fog, seed=fi+100)
                count = simple_cluster_count(deg)
                rate  = min(count / baseline, 1.0) \
                        if baseline > 0 else 0.0
                scores[fogi, ri] += rate

    scores /= max(len(frame_paths), 1)
    return scores, rain_levels, fog_levels


def plot_safety_heatmap(scores, rain_levels, fog_levels):
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    colors_list = [
        (0.0,  '#8B0000'),
        (0.70, '#FF4500'),
        (0.82, '#FFD700'),
        (0.87, '#90EE90'),
        (1.0,  '#006400'),
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'safety', [(v, c) for v, c in colors_list]
    )

    im = ax.imshow(scores, cmap=cmap, vmin=0.5, vmax=1.0,
                   aspect='auto', origin='upper')

    ax.set_xticks(range(len(rain_levels)))
    ax.set_xticklabels(
        [f"{r}\nmm/hr" for r in rain_levels],
        color='white', fontsize=10
    )
    ax.set_yticks(range(len(fog_levels)))
    ax.set_yticklabels(
        [f"{v}m" for v in fog_levels],
        color='white', fontsize=10
    )
    ax.set_xlabel('Rain Intensity (mm/hr)',
                  color='white', fontsize=12)
    ax.set_ylabel('Fog Visibility (m)',
                  color='white', fontsize=12)
    ax.set_title(
        'LiDAR Safety Heatmap — Operational Design Domain\n'
        'Vamshikrishna Gadde  |  MS Robotics ASU  |  Day 7 of 90',
        color='white', fontsize=13
    )

    for i in range(len(fog_levels)):
        for j in range(len(rain_levels)):
            score  = scores[i, j]
            color  = 'white' if score < 0.88 else 'black'
            label  = f"{score*100:.0f}%"
            status = "SAFE" if score >= SAFETY_THRESHOLD \
                     else "UNSAFE"
            s_color = '#00FF88' if score >= SAFETY_THRESHOLD \
                      else '#FF4444'
            ax.text(j, i - 0.15, label,
                    ha='center', va='center',
                    color=color, fontsize=9,
                    fontweight='bold')
            ax.text(j, i + 0.22, status,
                    ha='center', va='center',
                    color=s_color, fontsize=7,
                    fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Detection Rate', color='white', fontsize=11)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    ax.text(0.02, 0.02,
            f'SAFE   = >= {SAFETY_THRESHOLD*100:.0f}% detection\n'
            f'UNSAFE = <  {SAFETY_THRESHOLD*100:.0f}% detection',
            transform=ax.transAxes,
            color='white', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#333',
                      alpha=0.8))

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, 'safety_heatmap.png')
    plt.savefig(path, dpi=130, bbox_inches='tight',
                facecolor='#1a1a1a')
    plt.close()
    print(f"  Saved: {path}")
    return path


def create_degradation_gif(frame_path):
    pts = load_kitti_lidar(frame_path)

    scenarios = [
        (pts,
         "Clean — 0 mm/hr, clear sky",    '#00FF88', "SAFE"),
        (simulate_rain(pts, 25)[0],
         "Light Rain — 25 mm/hr",          '#00C8FF', "SAFE"),
        (simulate_rain(pts, 100)[0],
         "Heavy Rain — 100 mm/hr",         '#0055FF', "SAFE"),
        (simulate_fog(pts, 75)[0],
         "Moderate Fog — 75m visibility",  '#FF9500', "SAFE"),
        (simulate_fog(pts, 50)[0],
         "Dense Fog — 50m visibility",     '#FF4400', "UNSAFE"),
        (simulate_fog(pts, 25)[0],
         "Very Dense Fog — 25m visibility",'#CC0000',  "UNSAFE"),
    ]

    print(f"\n  Building animated GIF...")
    print(f"  {len(scenarios)} frames")

    frames    = []
    max_range = 45

    for i, (cloud, title, color, status) in \
            enumerate(scenarios):
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#0d0d0d')

        mask     = np.sqrt(
            (cloud[:, :2]**2).sum(axis=1)
        ) < max_range
        pts_show = cloud[mask]

        ax.scatter(pts_show[:, 0], pts_show[:, 1],
                   s=0.3, c=color, alpha=0.5)

        for r in [10, 20, 30, 40]:
            circle = plt.Circle(
                (0, 0), r, fill=False,
                color='#333', linewidth=0.8
            )
            ax.add_patch(circle)
            ax.text(0, r + 0.5, f'{r}m',
                    ha='center', color='#555', fontsize=7)

        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_aspect('equal')
        ax.tick_params(colors='#555')
        for sp in ax.spines.values():
            sp.set_edgecolor('#333')

        ax.set_title(
            "LiDAR Point Cloud — Weather Degradation\n"
            "Vamshikrishna Gadde  |  MS Robotics ASU",
            color='white', fontsize=10, pad=10
        )
        ax.text(0.5, 0.97, title,
                transform=ax.transAxes,
                ha='center', va='top',
                color='white', fontsize=11,
                fontweight='bold')
        ax.text(0.5, 0.92,
                f"Points: {len(pts_show):,}",
                transform=ax.transAxes,
                ha='center', va='top',
                color='#AAAAAA', fontsize=9)

        s_color = '#00FF88' if status == "SAFE" else '#FF3300'
        ax.text(0.5, 0.06, status,
                transform=ax.transAxes,
                ha='center', va='bottom',
                color=s_color, fontsize=16,
                fontweight='bold')

        ax.text(0.97, 0.03,
                f"Frame {i+1}/{len(scenarios)}",
                transform=ax.transAxes,
                ha='right', color='#555', fontsize=8)

        tmp = os.path.join(RESULTS_DIR,
                           f'_tmp_frame_{i}.png')
        plt.savefig(tmp, dpi=100, bbox_inches='tight',
                    facecolor='#0d0d0d')
        plt.close()
        frames.append(imageio.imread(tmp))
        print(f"    Frame {i+1}: {title[:45]}")

    gif_path  = os.path.join(RESULTS_DIR,
                              'weather_degradation.gif')
    durations = [1.5, 1.0, 1.5, 1.0, 1.5, 2.0]

    imageio.mimsave(gif_path, frames,
                    duration=durations, loop=0)

    for i in range(len(scenarios)):
        tmp = os.path.join(RESULTS_DIR,
                           f'_tmp_frame_{i}.png')
        if os.path.exists(tmp):
            os.remove(tmp)

    print(f"  Saved: {gif_path}")
    return gif_path


if __name__ == "__main__":

    print("\n" + "="*58)
    print("  Weather Visualization — Heatmap + GIF")
    print("  Day 7 of 90")
    print("="*58)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    files = sorted([
        os.path.join(KITTI_DIR, f)
        for f in os.listdir(KITTI_DIR)
        if f.endswith('.bin')
    ])[:5]

    print(f"\n  Frames for heatmap : {len(files)}")

    scores, rain_levels, fog_levels = \
        build_safety_heatmap(files)
    heatmap_path = plot_safety_heatmap(
        scores, rain_levels, fog_levels
    )

    gif_frame = files[4] if len(files) > 4 else files[0]
    print(f"\n  GIF frame: {os.path.basename(gif_frame)}")
    gif_path = create_degradation_gif(gif_frame)

    print(f"\n  {'='*58}")
    print(f"  COMPLETE")
    print(f"  Heatmap : {heatmap_path}")
    print(f"  GIF     : {gif_path}")
    print(f"\n  Open results/ to see both files.")
    print("="*58)