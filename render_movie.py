"""Render the BILGE-HOLE Tecplot history as a Motion-JPEG AVI."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.patches import Polygon

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ZONE_SIZE = re.compile(r"\bI\s*=\s*(\d+)", re.IGNORECASE)


def read_zones(path: Path) -> list[np.ndarray]:
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    zones: list[np.ndarray] = []
    index = 0
    while index < len(lines):
        match = ZONE_SIZE.search(lines[index])
        if match is None:
            index += 1
            continue
        count = int(match.group(1))
        values = [list(map(float, row.split())) for row in lines[index + 1 : index + count + 1]]
        if len(values) != count or any(len(row) != 4 for row in values):
            raise ValueError(f"Incomplete zone near line {index + 1} in {path}")
        zones.append(np.asarray(values, dtype=float))
        index += count + 1
    return zones


def render(data_dir: Path, output: Path, ffmpeg: Path, fps: int) -> None:
    zones = read_zones(data_dir / "NOZZLE1.plt")
    if len(zones) % 2:
        raise ValueError(f"Expected bubble/surface zone pairs, found {len(zones)} zones")

    bubbles = zones[0::2]
    surfaces = zones[1::2]
    history = np.loadtxt(data_dir / "RATIO-DTIME.TXT")
    if history.ndim == 1:
        history = history[None, :]
    if len(history) == len(bubbles) + 1:
        history = history[1:]
    if len(history) != len(bubbles):
        raise ValueError(f"Found {len(bubbles)} frames but {len(history)} history rows")

    width, height, dpi = 1280, 720, 100
    background = "#07111b"
    cyan = "#34d7e8"
    gold = "#ffc857"
    muted = "#8294a8"

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=background)
    ax = fig.add_axes((0.06, 0.10, 0.68, 0.82), facecolor=background)
    chart = fig.add_axes((0.78, 0.18, 0.19, 0.25), facecolor="#0b1824")

    wall = surfaces[0]
    ax.plot(wall[:, 0], wall[:, 1], color="#aab4bf", linewidth=5.0, alpha=0.55)
    ax.plot(wall[:, 0], wall[:, 1], color="#e2e8ee", linewidth=1.2, alpha=0.9)
    bubble = bubbles[0]
    patch = Polygon(bubble[:, :2], closed=True, facecolor=cyan, edgecolor="#baf7ff", alpha=0.40, linewidth=2.2)
    ax.add_patch(patch)

    stride = 4
    vectors = ax.quiver(
        bubble[::stride, 0], bubble[::stride, 1], bubble[::stride, 2], bubble[::stride, 3],
        color=gold, angles="xy", scale_units="xy", scale=12.0, width=0.0032,
        headwidth=4.2, headlength=5.5, alpha=0.9,
    )

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.25, 0.55)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"radial position  $R/R_m$", color="#cbd5df", fontsize=11)
    ax.set_ylabel(r"axial position  $Z/R_m$", color="#cbd5df", fontsize=11)
    ax.tick_params(colors=muted, labelsize=9)
    ax.grid(color="#274052", linewidth=0.6, alpha=0.34)
    for spine in ax.spines.values():
        spine.set_color("#456173")
    ax.set_title("Cavitation bubble collapse near a bilge hole", color="white", fontsize=17, loc="left", pad=14)

    status = fig.text(0.78, 0.83, "", color="white", fontsize=12, va="top", family="monospace")
    fig.text(0.78, 0.50, "Normalized bubble volume", color="#cbd5df", fontsize=10)
    chart.plot(history[:, 0], history[:, 1], color="#39586a", linewidth=1.2)
    progress, = chart.plot([], [], color=cyan, linewidth=2.0)
    marker, = chart.plot([], [], "o", color=gold, markersize=5)
    chart.set_xlim(history[0, 0], history[-1, 0])
    chart.set_ylim(0.0, max(1.02, float(history[:, 1].max()) * 1.02))
    chart.set_xlabel("time (ms)", color=muted, fontsize=8)
    chart.tick_params(colors=muted, labelsize=7)
    chart.grid(color="#274052", linewidth=0.5, alpha=0.4)
    for spine in chart.spines.values():
        spine.set_color("#456173")

    command = [
        str(ffmpeg), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "mjpeg",
        "-q:v", "3", "-pix_fmt", "yuvj420p", str(output),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for index, (contour, row) in enumerate(zip(bubbles, history, strict=True)):
            patch.set_xy(contour[:, :2])
            vectors.set_offsets(contour[::stride, :2])
            vectors.set_UVC(contour[::stride, 2], contour[::stride, 3])
            progress.set_data(history[: index + 1, 0], history[: index + 1, 1])
            marker.set_data([row[0]], [row[1]])
            speed = np.hypot(contour[:, 2], contour[:, 3]).max()
            status.set_text(
                f"STEP   {index + 1:03d} / {len(bubbles):03d}\n"
                f"TIME   {row[0]:7.1f} ms\n"
                f"V/V0   {row[1]:7.4f}\n"
                f"|U|max {speed:7.4f}"
            )
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
            process.stdin.write(frame.tobytes())
            if (index + 1) % 25 == 0 or index + 1 == len(bubbles):
                print(f"Rendered {index + 1}/{len(bubbles)} frames", flush=True)
    finally:
        process.stdin.close()
        return_code = process.wait()
        plt.close(fig)
    if return_code:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("run/full"))
    parser.add_argument("--output", type=Path, default=Path("bilge-hole-collapse.avi"))
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    render(args.data_dir.resolve(), args.output.resolve(), args.ffmpeg.resolve(), args.fps)


if __name__ == "__main__":
    main()
