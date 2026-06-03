# rain_simulator.py
# Simulates rain and fog on LiDAR point clouds
# Uses Marshall-Palmer attenuation model for rain
# Uses Koschmieder visibility law for fog
#
# Why this matters:
#   AV companies test in sunny weather and deploy in rain.
#   This measures exactly where the system breaks.
#   That threshold is what safety engineers need.
#
# Vamshikrishna Gadde | Day 7 of 90 | MS Robotics ASU

import numpy as np
import os


# LiDAR specs — KITTI uses Velodyne HDL-64E at 905nm
MAX_RANGE  = 120.0   # max detection range in meters
ALPHA_RAIN = 0.01    # rain attenuation coeff (905nm, Marshall-Palmer)
ALPHA_FOG  = 0.05    # fog attenuates much harder — smaller droplets


def load_kitti_lidar(filepath):
    # KITTI stores point clouds as binary float32
    # each point is (x, y, z, intensity)
    pts = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
    return pts


def get_ranges(points):
    # euclidean distance from sensor origin to each point
    return np.sqrt((points[:, :3] ** 2).sum(axis=1))


def simulate_rain(points, intensity_mmhr, seed=42):
    """
    Degrade a clean LiDAR scan to simulate rain.

    Three physical effects modeled:

    1. Attenuation
       Rain absorbs and scatters laser pulses.
       Probability of a point surviving drops
       with rain intensity and point range.
       Heavy rain + long range = point vanishes.

    2. Position noise
       Surviving points shift slightly.
       Caused by beam deflection through droplets.

    3. Backscatter (ghost points)
       Rain droplets reflect laser back early.
       Creates false points near the sensor (< 15m).
       These are the dangerous ghost detections.

    Rain intensity scale (mm/hr):
       0   = clear sky
       10  = light rain  (drizzle)
       25  = moderate rain
       50  = heavy rain  (monsoon level)
       100 = extreme rain

    Returns degraded point cloud + stats dict.
    """
    rng = np.random.RandomState(seed)

    if intensity_mmhr == 0:
        return points.copy(), {
            'original':      len(points),
            'survived':      len(points),
            'removed':       0,
            'ghost':         0,
            'survival_rate': 1.0
        }

    ranges = get_ranges(points)

    # survival probability — exponential decay with range and intensity
    # distant points in heavy rain almost never return
    p_survive = np.exp(
        -ALPHA_RAIN * intensity_mmhr * ranges / MAX_RANGE
    )
    mask      = rng.random(len(points)) < p_survive
    survived  = points[mask].copy()
    n_removed = int((~mask).sum())

    # add position noise to surviving points
    # scales with rain intensity and range
    noise_std = 0.01 * (intensity_mmhr / 100.0) * \
                (ranges[mask] / MAX_RANGE)
    survived[:, :3] += rng.randn(len(survived), 3) * \
                       noise_std[:, None]

    # backscatter ghost points near sensor
    n_ghost = int(0.002 * intensity_mmhr * len(points))
    ghost_cloud = np.zeros((0, 4))

    if n_ghost > 0:
        r  = rng.uniform(1.0, 15.0, n_ghost)
        az = rng.uniform(-np.pi, np.pi, n_ghost)
        el = rng.uniform(-0.3, 0.3, n_ghost)

        gx = r * np.cos(el) * np.cos(az)
        gy = r * np.cos(el) * np.sin(az)
        gz = r * np.sin(el)
        gi = rng.uniform(0.0, 0.3, n_ghost)

        ghost_cloud = np.column_stack([gx, gy, gz, gi])

    if n_ghost > 0:
        result = np.vstack([survived, ghost_cloud])
    else:
        result = survived

    stats = {
        'original':      len(points),
        'survived':      len(survived),
        'removed':       n_removed,
        'ghost':         n_ghost,
        'survival_rate': len(survived) / len(points)
    }

    return result, stats


def simulate_fog(points, visibility_m, seed=42):
    """
    Degrade a clean LiDAR scan to simulate fog.

    Fog is worse than rain for LiDAR.
    Smaller droplets = stronger attenuation per meter.
    Uses Koschmieder's law:
      extinction = 3.912 / visibility_m

    Fog also creates a dense haze near the sensor —
    a cloud of false returns that masks real objects.

    Visibility scale (meters):
       200 = light fog  (can see far)
       100 = moderate fog
       50  = dense fog  (difficult driving)
       25  = very dense
       10  = extreme fog (near zero visibility)

    Returns degraded point cloud + stats dict.
    """
    rng    = np.random.RandomState(seed)
    ranges = get_ranges(points)

    # Koschmieder extinction — stronger than rain
    extinction = 3.912 / max(visibility_m, 1.0)
    p_survive  = np.exp(-extinction * ranges / 10.0)
    mask       = rng.random(len(points)) < p_survive
    survived   = points[mask].copy()
    n_removed  = int((~mask).sum())

    # fog haze — dense false returns near sensor
    fog_fraction = max(0.0, 1.0 - visibility_m / 200.0)
    n_haze       = int(0.1 * fog_fraction * len(points))
    n_haze       = min(n_haze, len(points) // 2)

    haze_cloud = np.zeros((0, 4))

    if n_haze > 0:
        r  = np.clip(
            rng.exponential(visibility_m / 10.0, n_haze),
            0.5, visibility_m
        )
        az = rng.uniform(-np.pi, np.pi, n_haze)
        el = rng.uniform(-0.2, 0.2, n_haze)

        hx = r * np.cos(el) * np.cos(az)
        hy = r * np.cos(el) * np.sin(az)
        hz = r * np.sin(el)
        hi = rng.uniform(0.0, 0.2, n_haze)

        haze_cloud = np.column_stack([hx, hy, hz, hi])

    if n_haze > 0:
        result = np.vstack([survived, haze_cloud])
    else:
        result = survived

    stats = {
        'original':      len(points),
        'survived':      len(survived),
        'removed':       n_removed,
        'haze':          n_haze,
        'survival_rate': len(survived) / len(points)
    }

    return result, stats


if __name__ == "__main__":

    KITTI_DIR = (
        r"C:\Users\vamsh\Downloads\kitti"
        r"\2011_09_26_drive_0001_sync"
        r"\2011_09_26"
        r"\2011_09_26_drive_0001_sync"
        r"\velodyne_points\data"
    )

    print("\n" + "="*58)
    print("  Rain + Fog Simulator — Day 7 of 90")
    print("  Testing on real KITTI LiDAR data")
    print("="*58)

    frame_path = os.path.join(KITTI_DIR, "0000000000.bin")
    points     = load_kitti_lidar(frame_path)

    print(f"\n  Clean point cloud : {len(points):,} points")
    print(f"  Max range         : {MAX_RANGE}m")
    print(f"  Sensor            : Velodyne HDL-64E")

    # rain benchmark
    print(f"\n  RAIN SIMULATION")
    print(f"  {'─'*58}")
    print(f"  {'mm/hr':<12} {'Survived':<12} {'Removed':<12}"
          f" {'Ghost':<10} {'Rate':<8} {'Weather'}")
    print(f"  {'─'*58}")

    rain_levels = [
        (0,   "Clear sky"),
        (10,  "Light rain"),
        (25,  "Moderate rain"),
        (50,  "Heavy rain"),
        (100, "Extreme rain"),
    ]

    for intensity, label in rain_levels:
        deg, s = simulate_rain(points, intensity)
        print(
            f"  {intensity:<12} "
            f"{s['survived']:<12,} "
            f"{s['removed']:<12,} "
            f"{s['ghost']:<10,} "
            f"{s['survival_rate']*100:<8.1f}% "
            f"{label}"
        )

    # fog benchmark
    print(f"\n  FOG SIMULATION")
    print(f"  {'─'*50}")
    print(f"  {'Vis (m)':<12} {'Survived':<12} "
          f"{'Haze pts':<12} {'Rate':<8} {'Condition'}")
    print(f"  {'─'*50}")

    fog_levels = [
        (200, "Light fog"),
        (100, "Moderate fog"),
        (50,  "Dense fog"),
        (25,  "Very dense fog"),
        (10,  "Extreme fog"),
    ]

    for vis, label in fog_levels:
        deg, s = simulate_fog(points, vis)
        print(
            f"  {vis:<12} "
            f"{s['survived']:<12,} "
            f"{s['haze']:<12,} "
            f"{s['survival_rate']*100:<8.1f}% "
            f"{label}"
        )

    print(f"\n  Simulator ready.")
    print(f"  Run weather_benchmark.py for full analysis.")
    print("="*58)