# analyze.py
# Visualizes what rain and fog actually do to a LiDAR scan
# Side by side comparison: clean vs degraded
# Also shows per-range analysis and ghost point distribution
#
# Nani | Day 7 of 90 | MS Robotics ASU

import numpy as np
import os
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
RESULTS_DIR = "results"


def plot_topdown(ax, points, title, color, max_range=50):
    # top-down (bird's eye) view of point cloud
    # only show points within max_range meters
    mask = np.sqrt((points[:, :2]**2).sum(axis=1)) < max_range
    pts  = points[mask]
    ax.scatter(pts[:, 0], pts[:, 1],
               s=0.3, c=color, alpha=0.4)
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_title(title, color='white', fontsize=10)
    ax.set_facecolor('#111111')
    ax.tick_params(colors='white', labelsize=7)
    ax.set_aspect('equal')
    # draw range rings
    for r in [10, 25, 50]:
        circle = plt.Circle((0, 0), r, fill=False,
                             color='#333333', linewidth=0.5)
        ax.add_patch(circle)


def range_analysis(points_clean, points_rain,
                   points_fog, bins=10):
    # measure point survival rate per distance band
    # shows whether weather kills close or far objects first
    ranges_clean = np.sqrt((points_clean[:, :3]**2).sum(axis=1))
    ranges_rain  = np.sqrt((points_rain[:, :3]**2).sum(axis=1))
    ranges_fog   = np.sqrt((points_fog[:, :3]**2).sum(axis=1))

    edges   = np.linspace(0, 80, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    clean_counts = np.histogram(ranges_clean, bins=edges)[0]
    rain_counts  = np.histogram(ranges_rain,  bins=edges)[0]
    fog_counts   = np.histogram(ranges_fog,   bins=edges)[0]

    # survival rate per distance band
    rain_survival = np.where(
        clean_counts > 0,
        rain_counts / clean_counts, 0
    )
    fog_survival = np.where(
        clean_counts > 0,
        fog_counts / clean_counts, 0
    )

    return centers, rain_survival, fog_survival


def create_analysis():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    frame_path = os.path.join(KITTI_DIR, "0000000005.bin")
    clean      = load_kitti_lidar(frame_path)

    # generate weather variants for comparison
    rain_light,  _ = simulate_rain(clean, 25)
    rain_heavy,  _ = simulate_rain(clean, 100)
    fog_moderate,_ = simulate_fog(clean, 75)
    fog_dense,   _ = simulate_fog(clean, 25)

    print(f"  Clean     : {len(clean):,} points")
    print(f"  Rain 25   : {len(rain_light):,} points")
    print(f"  Rain 100  : {len(rain_heavy):,} points")
    print(f"  Fog 75m   : {len(fog_moderate):,} points")
    print(f"  Fog 25m   : {len(fog_dense):,} points")

    # figure 1 — top-down comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('#0d0d0d')
    fig.suptitle(
        "LiDAR Point Cloud Degradation — Rain vs Fog\n"
        "Vamshikrishna Gadde  |  MS Robotics ASU  |  Day 7 of 90",
        fontsize=13, color='white'
    )

    scenarios = [
        (clean,         "Clean (0 mm/hr, clear)",    '#00FF88'),
        (rain_light,    "Light Rain (25 mm/hr)",     '#00C8FF'),
        (rain_heavy,    "Heavy Rain (100 mm/hr)",    '#0066FF'),
        (clean,         "Clean (200m visibility)",   '#00FF88'),
        (fog_moderate,  "Moderate Fog (75m vis)",    '#FF9500'),
        (fog_dense,     "Dense Fog (25m vis)",       '#FF3300'),
    ]

    for ax, (pts, title, color) in zip(axes.flat, scenarios):
        plot_topdown(ax, pts, title, color)

    # add point count annotations
    for ax, (pts, _, _) in zip(axes.flat, scenarios):
        ax.text(0.02, 0.98, f"{len(pts):,} pts",
                transform=ax.transAxes,
                color='white', fontsize=8,
                va='top')

    plt.tight_layout()
    p1 = os.path.join(RESULTS_DIR, 'pointcloud_comparison.png')
    plt.savefig(p1, dpi=130, bbox_inches='tight',
                facecolor='#0d0d0d')
    plt.close()
    print(f"  Saved: {p1}")

    # figure 2 — per-range survival analysis
    centers, rain_surv, fog_surv = range_analysis(
        clean, rain_heavy, fog_dense
    )

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
    fig2.patch.set_facecolor('#1a1a1a')
    fig2.suptitle(
        "Point Survival Rate by Distance Band\n"
        "Vamshikrishna Gadde  |  MS Robotics ASU  |  Day 7 of 90",
        fontsize=12, color='white'
    )

    def style(ax):
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#444')

    ax1 = axes2[0]
    style(ax1)
    ax1.bar(centers, rain_surv * 100,
            width=6, color='#00C8FF',
            edgecolor='#333', alpha=0.85)
    ax1.axhline(y=85, color='red', linestyle='--',
                linewidth=1.5, label='85% safety limit')
    ax1.set_xlabel('Distance from Sensor (m)', color='white')
    ax1.set_ylabel('Point Survival Rate (%)', color='white')
    ax1.set_title('Rain 100 mm/hr — Survival by Range',
                  color='white', fontsize=11)
    ax1.set_ylim(0, 120)
    ax1.legend(facecolor='#1a1a1a', labelcolor='white')
    for i, (c, v) in enumerate(zip(centers, rain_surv)):
        ax1.text(c, v*100 + 2, f'{v*100:.0f}%',
                 ha='center', color='white', fontsize=8)

    ax2 = axes2[1]
    style(ax2)
    ax2.bar(centers, fog_surv * 100,
            width=6, color='#FF6B35',
            edgecolor='#333', alpha=0.85)
    ax2.axhline(y=85, color='red', linestyle='--',
                linewidth=1.5, label='85% safety limit')
    ax2.set_xlabel('Distance from Sensor (m)', color='white')
    ax2.set_ylabel('Point Survival Rate (%)', color='white')
    ax2.set_title('Fog 25m visibility — Survival by Range',
                  color='white', fontsize=11)
    ax2.set_ylim(0, 120)
    ax2.legend(facecolor='#1a1a1a', labelcolor='white')
    for i, (c, v) in enumerate(zip(centers, fog_surv)):
        ax2.text(c, v*100 + 2, f'{v*100:.0f}%',
                 ha='center', color='white', fontsize=8)

    plt.tight_layout()
    p2 = os.path.join(RESULTS_DIR, 'range_analysis.png')
    plt.savefig(p2, dpi=130, bbox_inches='tight',
                facecolor='#1a1a1a')
    plt.close()
    print(f"  Saved: {p2}")

    return p1, p2


if __name__ == "__main__":

    print("\n" + "="*55)
    print("  Point Cloud Weather Analysis")
    print("  Day 7 of 90")
    print("="*55)

    p1, p2 = create_analysis()

    print(f"\n  Results:")
    print(f"  {p1}")
    print(f"  {p2}")
    print(f"\n  Open both images to see the degradation.")
    print("="*55)