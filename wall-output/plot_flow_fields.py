from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.path import Path as MplPath
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import distance_transform_edt


SOURCE = Path(__file__).with_name("WALL-FLOW-FIELD.plt")
BUBBLE_SOURCE = Path(__file__).with_name("WALL-CAVITATION.plt")
SELECTED_STEPS = {50, 100, 200, 310}
GRID_SHAPE = (161, 161)


def read_selected_zones(path):
    zones = {}
    with path.open("r", encoding="ascii") as stream:
        while line := stream.readline():
            if not line.startswith("ZONE"):
                continue
            step = int(re.search(r"K=\s*(\d+)", line).group(1))
            time_value = float(re.search(r"SOLUTIONTIME=\s*([^\s]+)", line).group(1))
            aux = stream.readline()
            rows = [stream.readline() for _ in range(GRID_SHAPE[0] * GRID_SHAPE[1])]
            if step in SELECTED_STEPS:
                zones[step] = (time_value, np.loadtxt(rows))
    return zones


def read_bubble_boundaries(path):
    boundaries = {}
    bubble_zone = True
    step = 0
    with path.open("r", encoding="ascii") as stream:
        while line := stream.readline():
            match = re.search(r"ZONE\s+I=\s*(\d+)", line)
            if not match:
                continue
            count = int(match.group(1))
            rows = [stream.readline() for _ in range(count)]
            if bubble_zone:
                step += 1
                if step in SELECTED_STEPS:
                    boundaries[step] = np.loadtxt(rows)[:, :2]
            bubble_zone = not bubble_zone
    return boundaries


def smooth_boundary(points):
    distance = np.r_[0.0, np.cumsum(np.hypot(*np.diff(points, axis=0).T))]
    parameter = distance / distance[-1]
    fine = np.linspace(0.0, 1.0, 800)
    radial = PchipInterpolator(parameter, points[:, 0])(fine)
    vertical = PchipInterpolator(parameter, points[:, 1])(fine)
    radial[[0, -1]] = 0.0
    return np.column_stack((np.maximum(radial, 0.0), vertical))


def pressure_limits(values):
    gauge = values - 1.0
    span = max(abs(np.nanpercentile(gauge, 2)), abs(np.nanpercentile(gauge, 98)), 0.05)
    return 1.0 - span, 1.0 + span


def plot_zone(step, time_value, raw, bubble_nodes):
    nz, nr = GRID_SHAPE
    fields = raw.reshape(nz, nr, 9)
    r, z = fields[..., 0], fields[..., 1]
    ur, uz, pstar = fields[..., 3], fields[..., 4], fields[..., 6]
    liquid = fields[..., 8] > 0.5

    pressure = np.where(liquid, pstar, np.nan)
    speed = np.where(liquid, np.hypot(ur, uz), np.nan)
    low, high = pressure_limits(pressure)
    norm = TwoSlopeNorm(vmin=low, vcenter=1.0, vmax=high)
    nearest = distance_transform_edt(~liquid, return_distances=False, return_indices=True)
    pressure_display = pressure[tuple(nearest)]
    bubble = smooth_boundary(bubble_nodes)
    bubble_path = MplPath(np.vstack((bubble, bubble[0])), closed=True)

    fig, ax = plt.subplots(figsize=(8.2, 7.1), constrained_layout=True)
    contour = ax.contourf(r, z, pressure_display, levels=np.linspace(low, high, 101),
                          cmap="coolwarm", norm=norm, extend="both")
    ax.fill(bubble[:, 0], bubble[:, 1], facecolor="white", edgecolor="none", zorder=3)
    ax.plot(bubble[:, 0], bubble[:, 1], color="black", linewidth=1.8, zorder=4)
    ax.axhline(0.0, color="black", linewidth=1.8, zorder=4)

    stride = 8
    sample = np.s_[::stride, ::stride]
    sample_points = np.column_stack((r[sample].ravel(), z[sample].ravel()))
    outside_bubble = ~bubble_path.contains_points(sample_points).reshape(r[sample].shape)
    vector_ok = liquid[sample] & outside_bubble
    rs, zs = r[sample][vector_ok], z[sample][vector_ok]
    urs, uzs = ur[sample][vector_ok], uz[sample][vector_ok]
    reference = max(np.nanpercentile(speed, 95), 1.0e-12)
    scale = 9.0 * reference
    vectors = ax.quiver(rs, zs, urs, uzs, color="#172554", angles="xy",
                        scale_units="xy", scale=scale, width=0.0032,
                        headwidth=3.5, headlength=4.5, minlength=0.05, zorder=5)

    cbar = fig.colorbar(contour, ax=ax, pad=0.025)
    cbar.set_label(r"dimensionless pressure $p^*$")
    ax.quiverkey(vectors, 0.73, 0.965, reference,
                 rf"$|\mathbf{{u}}|={reference:.3g}$", labelpos="E",
                 coordinates="axes", color="#172554")
    ax.set_title(rf"Exterior flow field: $K={step}$, $t^*={time_value:.6f}$")
    ax.set_xlabel(r"radial coordinate $r/R_0$")
    ax.set_ylabel(r"height above wall $z/R_0$")
    ax.set_aspect("equal")
    ax.set_xlim(float(np.nanmin(r)), float(np.nanmax(r)))
    ax.set_ylim(float(np.nanmin(z)), float(np.nanmax(z)))
    ax.grid(color="black", alpha=0.10, linewidth=0.5)
    ax.text(0.99, 0.015, "black curve: BEM bubble surface",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})

    target = Path(__file__).with_name(f"flow-pressure-vector-k{step:03d}.png")
    fig.savefig(target, dpi=180)
    plt.close(fig)
    print(f"{target.name}: p* robust range [{low:.6g}, {high:.6g}], "
          f"max |u|={np.nanmax(speed):.6g}")


zones = read_selected_zones(SOURCE)
bubbles = read_bubble_boundaries(BUBBLE_SOURCE)
missing = SELECTED_STEPS.difference(zones).union(SELECTED_STEPS.difference(bubbles))
if missing:
    raise RuntimeError(f"Missing requested zones: {sorted(missing)}")
for selected_step in sorted(zones):
    plot_zone(selected_step, *zones[selected_step], bubbles[selected_step])
