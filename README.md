# Day 7 — Adverse Weather LiDAR Perception Analysis

> **Series 1: Perception | Project 7 of 12**
> MS Robotics & Autonomous Systems Engineering — Arizona State University — Dec 2026

---

## The Question Nobody Answers

AV companies test in perfect weather. Then they deploy in rain and fog.
Systems fail. The question is: **at exactly what rainfall or fog density does your LiDAR become unsafe?**

Waymo has this number. They do not publish it.
I measured mine on real KITTI sensor data.

---

## Animated Point Cloud Degradation

*Watch the LiDAR scan degrade from clean to fog — 6 weather conditions*

![Weather Degradation GIF](https://drive.google.com/uc?id=17AtRqY8vLZ9kbbKmVphnm3AjdLI75TJk)

---

## Safety Heatmap — Operational Design Domain

*Every rain + fog combination tested. Green = safe. Red = unsafe.*

![Safety Heatmap](https://drive.google.com/uc?id=19oRWBZwNUl0o1AJBHXjUf_JeLJT9_VvO)

**The finding:** Almost the entire heatmap is green.
Only 10m visibility fog causes unsafe detection.
LiDAR is far more weather-resistant than most people assume.

---

## Detection Accuracy vs Weather Intensity

![Weather Benchmark](https://drive.google.com/uc?id=17xpAMMFAVRLhQkHCofqGgV7qf8YJe3Xi)

---

## Point Cloud Comparison — Clean vs Rain vs Fog

*Top row: rain progression. Bottom row: fog progression.*

![Point Cloud Comparison](https://drive.google.com/uc?id=1kDYBJiTIdQSZO-LMa_jXfdVLmVWBbF-H)

---

## Per-Range Survival Analysis

*How weather affects close vs distant objects differently*

![Range Analysis](https://drive.google.com/uc?id=1L-PDGuS4Ue0m2Wsv4Ws1jBFZxTHtXoCv)

---

## Key Findings

### Finding 1 — Rain Does Not Kill LiDAR

| Rain (mm/hr) | Detection Rate | Status |
|---|---|---|
| 0 (clear) | 98.5% | Safe |
| 10 (light) | 98.7% | Safe |
| 25 (moderate) | 93.3% | Safe |
| 50 (heavy) | 89.5% | Safe |
| 75 (very heavy) | 91.8% | Safe |
| 100 (extreme) | 88.8% | Safe |

Rain at 100 mm/hr — Phoenix monsoon level — still achieves 88.8% detection.
LiDAR survives rain because laser pulses at 905nm wavelength are not significantly
absorbed by millimeter-scale water droplets. This is exactly why Waymo chose
LiDAR over cameras for adverse weather operation.

### Finding 2 — Fog is the Real Threat

| Visibility (m) | Detection Rate | Status |
|---|---|---|
| 200 (light fog) | 93.2% | Safe |
| 150 | 94.1% | Safe |
| 100 (moderate) | 89.4% | Safe |
| 75 | 87.5% | Safe |
| 50 (dense) | 80.0% | **UNSAFE** |
| 25 (very dense) | 59.5% | **UNSAFE** |
| 10 (extreme) | 50.5% | **UNSAFE** |

The safety boundary is between 75m and 50m visibility.
Below 75m visibility this LiDAR system cannot be trusted for safe operation.
Fog uses Koschmieder extinction — fine droplets attenuate laser energy
far more aggressively than rain droplets across the same distance.

### Finding 3 — Rain Adds Ghost Points, Not Misses

```
Clean cloud      : 121,948 points
Rain 25 mm/hr    : 124,458 points  (+2,510 ghost points)
Rain 100 mm/hr   : 132,804 points  (+10,856 ghost points)
```

Rain does not remove real detections — it adds fake ones.
At 100 mm/hr there are 24,203 backscatter ghost points in the near field (< 15m).
These create false cluster detections that could trigger unnecessary emergency braking.
The danger from rain is not missed objects. It is false alarms.

### Finding 4 — Range Determines Rain Vulnerability

From the range analysis chart:
- Rain 100 mm/hr at 0-10m: **126% survival** (ghost points dominate)
- Rain 100 mm/hr at 30m: **80% survival** (below safety threshold)
- Rain 100 mm/hr at 60m+: **55-57% survival** (unreliable)

Rain kills distant objects first. A pedestrian at 40m in heavy rain
has only 69% point survival. At 5m the same pedestrian appears inflated
with fake backscatter points around them.

### Finding 5 — The ODD is Almost Unlimited

The operational design domain for this LiDAR system:

```
Rain:  Safe at any intensity tested (up to 100 mm/hr)
Fog:   Safe above 75m visibility
       Unsafe below 50m visibility

This covers:
  All normal driving rain         ✓
  Monsoon conditions              ✓
  Light and moderate fog          ✓
  Dense fog below 50m visibility  ✗
```

---

## What I Built

### Physics Models

**Rain — Marshall-Palmer Attenuation**
```
P_survive = exp(-alpha * intensity_mmhr * range / MAX_RANGE)
alpha = 0.01 (905nm Velodyne HDL-64E)
```

**Fog — Koschmieder Visibility Law**
```
extinction = 3.912 / visibility_m
P_survive  = exp(-extinction * range / 10.0)
```

**Backscatter** — rain droplets create false returns near sensor,
modeled proportional to intensity and point density.

### Pipeline

```
KITTI LiDAR frame (121k points)
        ↓
Weather simulation (rain / fog / combined)
        ↓
RANSAC ground removal
        ↓
Height filter (-1.5m to 3.0m)
        ↓
Grid-based clustering
        ↓
Detection count vs clean baseline
        ↓
Safety threshold analysis (85% baseline)
```

### Benchmark Scale

```
Frames tested        : 10 real KITTI sequences
Rain conditions      : 6 levels (0-100 mm/hr)
Fog conditions       : 7 levels (10-200m visibility)
Heatmap combinations : 42 rain x fog pairs
Total evaluations    : 130 per frame x 10 frames
```

---

## What I Learned

**Why fog beats rain for LiDAR degradation:**
Rain droplets are 1-5mm diameter. Fog droplets are 1-100 micrometers.
The smaller the droplet the more interaction per unit volume.
A cubic meter of dense fog contains millions more droplets than
a cubic meter of heavy rain — each one scattering laser energy.
This is why visibility range matters more than rainfall rate
when characterizing LiDAR performance.

**The ghost point problem is underappreciated:**
Most papers report missed detection rate from rain.
The backscatter ghost point rate gets less attention.
But false positives in a safety system are equally dangerous —
phantom braking at highway speed causes rear collisions.
Measuring ghost points separately from real point loss
gives a more complete picture of rain's actual danger.

**The ODD finding changes how you think about sensor choice:**
If rain never breaks LiDAR above 85% detection
but cameras fail beyond 35m even in clear weather (Day 2 finding),
the sensor fusion argument becomes very clear.
LiDAR handles weather. Cameras handle range texture and color.
Neither is sufficient alone. Both together cover each other's failures.
This is the exact argument behind every modern AV sensor suite.

---

## How This Connects to the Series

```
Day 2: Cameras fail beyond 35m in clear weather
Day 7: LiDAR fails below 25m visibility in fog
       LiDAR survives all rain conditions

Combined finding:
  Camera failure mode:  range-dependent in clear weather
  LiDAR failure mode:   extreme fog only

They fail in completely different conditions.
Fusing them covers both failure modes.
Day 8 builds that fusion pipeline.
```

---

## Run It Yourself

```bash
git clone https://github.com/GVK-Engine/day-007-adverse-weather-perception
cd day-007-adverse-weather-perception
pip install -r requirements.txt
```

```bash
# Test rain and fog simulation on one frame
py -3.11 rain_simulator.py

# Full benchmark across 10 frames x 13 conditions
py -3.11 weather_benchmark.py

# Point cloud comparison and range analysis
py -3.11 analyze.py

# Safety heatmap and animated GIF
py -3.11 visualize_weather.py
```

Update `KITTI_DIR` in each file to your local KITTI path.
KITTI download: https://www.cvlibs.net/datasets/kitti/raw_data.php

---

## Project Structure

```
day-007-adverse-weather-perception/
├── rain_simulator.py      Physics models for rain and fog
├── weather_benchmark.py   Detection accuracy vs weather intensity
├── analyze.py             Point cloud visualization and range analysis
├── visualize_weather.py   Safety heatmap and animated GIF
├── requirements.txt
└── results/
    ├── weather_benchmark.png
    ├── pointcloud_comparison.png
    ├── range_analysis.png
    ├── safety_heatmap.png
    └── weather_degradation.gif
```

---

## Stack

`Python 3.11` `NumPy` `Matplotlib` `imageio` `KITTI Velodyne HDL-64E`

---

## Series 1 — Perception Progress

| # | Project | Status |
|---|---------|--------|
| P1.1 | LiDAR Obstacle Detection | ✅ Complete |
| P1.2 | Stereo Camera Depth Analysis | ✅ Complete |
| P1.3 | PointPillars 3D Detector | ✅ Complete |
| P1.4 | Multi-Camera BEV Perception | ✅ Complete |
| P1.5 | Multi-Object Tracking SORT | ✅ Complete |
| P1.6 | Semantic Segmentation ROS2 | ✅ Complete |
| P1.7 | Adverse Weather Perception | ✅ Complete |
