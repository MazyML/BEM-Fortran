#!/usr/bin/env python3
"""Render the complete pre/post-collapse BEM run as a Motion-JPEG AVI."""

from __future__ import annotations

import io
import re
import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT, FPS = 1280, 720, 24
PLOT = (92, 88, 1200, 650)
YLIM = (-0.25, 2.75)
_plot_aspect = (PLOT[2] - PLOT[0]) / (PLOT[3] - PLOT[1])
_xspan = (YLIM[1] - YLIM[0]) * _plot_aspect
XLIM = (-_xspan / 2, _xspan / 2)


def read_zones(path: Path):
    lines = path.read_text().splitlines()
    zones, index = [], 0
    while index < len(lines):
        match = re.search(r"ZONE\s+I=\s*(\d+)", lines[index])
        if not match:
            index += 1
            continue
        count = int(match.group(1))
        points = [tuple(map(float, row.split()[:4])) for row in lines[index + 1 : index + count + 1]]
        if len(points) != count:
            raise ValueError(f"Incomplete zone near line {index + 1}")
        zones.append(points)
        index += count + 1
    return zones


def read_half_frames(path: Path):
    """Read +r bubble, -r bubble, +r wall, -r wall for each step."""
    zones = read_zones(path)
    if len(zones) % 4:
        raise ValueError("Expected four half-profile zones per accepted step")
    frames = []
    for index in range(0, len(zones), 4):
        positive, negative, wall_positive, wall_negative = zones[index : index + 4]
        if any(abs(point[1]) > 1e-10 for point in wall_positive + wall_negative):
            raise ValueError(f"Rigid-wall zone is not at z=0 near zone {index + 3}")
        frames.append((positive, negative))
    return frames


def read_history(path: Path):
    return [tuple(map(float, row.split()[:2])) for row in path.read_text().splitlines() if len(row.split()) >= 2]


def read_postcollapse(path: Path):
    rows = []
    for row in path.read_text().splitlines():
        fields = row.split()
        if not fields or fields[0] == "#":
            continue
        rows.append(tuple(map(float, fields)))
    return rows


def load_font(size: int, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def xy(point, scale=2):
    left, top, right, bottom = (value * scale for value in PLOT)
    x, y = point[:2]
    return (
        round(left + (x - XLIM[0]) / (XLIM[1] - XLIM[0]) * (right - left)),
        round(bottom - (y - YLIM[0]) / (YLIM[1] - YLIM[0]) * (bottom - top)),
    )


def vertical_text(image, text, text_font, scale=2):
    box = ImageDraw.Draw(image).textbbox((0, 0), text, font=text_font)
    label = Image.new("RGBA", (box[2] + 8 * scale, box[3] + 8 * scale), (0, 0, 0, 0))
    ImageDraw.Draw(label).text((4 * scale, 4 * scale), text, font=text_font, fill=(160, 181, 194))
    label = label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(label, (19 * scale, (HEIGHT * scale - label.height) // 2), label)


def render(positive, negative, step, count, time_value, ratio, centroid, ring=None):
    scale = 2
    image = Image.new("RGB", (WIDTH * scale, HEIGHT * scale), (7, 16, 29))
    draw = ImageDraw.Draw(image)
    title = load_font(30 * scale, True)
    body = load_font(18 * scale)
    small = load_font(15 * scale)
    left, top, right, bottom = (value * scale for value in PLOT)

    draw.text((92 * scale, 27 * scale), "Cavitation bubble collapse above a rigid wall", font=title, fill=(235, 245, 250))
    draw.text((1200 * scale, 38 * scale), "Axisymmetric BEM · γ = 1.5", font=body, fill=(126, 156, 178), anchor="ra")
    draw.rounded_rectangle((left, top, right, bottom), radius=12 * scale, fill=(11, 27, 43), outline=(47, 73, 92), width=2 * scale)

    for tick in (-2, -1, 0, 1, 2):
        a, b = xy((tick, YLIM[0]), scale), xy((tick, YLIM[1]), scale)
        draw.line((a, b), fill=(26, 48, 65), width=scale)
        draw.text((a[0], bottom + 11 * scale), str(tick), font=small, fill=(121, 151, 170), anchor="ma")
    for tick in (0, 0.5, 1, 1.5, 2, 2.5):
        a, b = xy((XLIM[0], tick), scale), xy((XLIM[1], tick), scale)
        draw.line((a, b), fill=(26, 48, 65), width=scale)
        draw.text((left - 12 * scale, a[1]), f"{tick:g}", font=small, fill=(121, 151, 170), anchor="rm")

    wall_y = xy((0, 0), scale)[1]
    draw.rectangle((left, wall_y, right, bottom), fill=(71, 79, 87))
    draw.line((left, wall_y, right, wall_y), fill=(207, 218, 224), width=4 * scale)
    draw.text((right - 15 * scale, wall_y + 14 * scale), "rigid wall · ∂φ/∂n = 0", font=small, fill=(214, 223, 228), anchor="ra")
    axis_a, axis_b = xy((0, 0), scale), xy((0, YLIM[1]), scale)
    draw.line((axis_a, axis_b), fill=(78, 109, 129), width=scale)

    positive_pixels = [xy(point, scale) for point in positive]
    negative_pixels = [xy(point, scale) for point in negative]
    if ring is None:
        polygon = positive_pixels + list(reversed(negative_pixels))
        draw.polygon(polygon, fill=(12, 158, 174))
        draw.line(polygon + [polygon[0]], fill=(151, 244, 244), width=3 * scale, joint="curve")
        phase = "pre-collapse · jet formation"
    else:
        # Closed signed-r contours must be drawn separately after reconnection.
        for polygon in (positive_pixels, negative_pixels):
            draw.polygon(polygon, fill=(12, 158, 174))
            draw.line(polygon + [polygon[0]], fill=(151, 244, 244), width=3 * scale, joint="curve")
        a_vr, c_vr = ring
        ring_left = xy((-a_vr, c_vr), scale)
        ring_right = xy((a_vr, c_vr), scale)
        ring_color = (244, 117, 204)
        draw.line((ring_left, ring_right), fill=ring_color, width=2 * scale)
        radius = 4 * scale
        for center in (ring_left, ring_right):
            draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=ring_color, outline=(255, 218, 244), width=scale)
        draw.text((ring_right[0] + 8 * scale, ring_right[1] - 3 * scale), "vortex ring", font=small, fill=ring_color, anchor="lm")
        phase = "post-collapse · toroidal bubble"

    draw.text((112 * scale, 110 * scale), f"step {step:03d} / {count}", font=body, fill=(221, 235, 242))
    draw.text((112 * scale, 139 * scale), f"t* = {time_value:.5f}", font=body, fill=(148, 187, 207))
    draw.text((112 * scale, 168 * scale), f"V / V0 = {ratio:.5f}", font=body, fill=(116, 225, 225))
    draw.text((112 * scale, 197 * scale), f"zc / R0 = {centroid:.5f}", font=body, fill=(177, 199, 211))
    draw.text((112 * scale, 226 * scale), phase, font=small, fill=(207, 171, 219))
    draw.text((650 * scale, 687 * scale), "radial coordinate r / R0", font=body, fill=(160, 181, 194), anchor="ma")
    vertical_text(image, "height above wall z / R0", body, scale)

    x0, y0, x1, y1 = 948 * scale, 116 * scale, 1168 * scale, 132 * scale
    draw.rounded_rectangle((x0, y0, x1, y1), radius=8 * scale, fill=(28, 52, 69))
    draw.rounded_rectangle((x0, y0, x0 + (x1 - x0) * step / count, y1), radius=8 * scale, fill=(21, 190, 198))
    draw.text((x1, 143 * scale), "simulation progress", font=small, fill=(127, 159, 176), anchor="ra")
    return image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def chunk(tag: bytes, payload: bytes):
    return tag + struct.pack("<I", len(payload)) + payload + (b"\0" if len(payload) & 1 else b"")


def list_chunk(kind: bytes, payload: bytes):
    return chunk(b"LIST", kind + payload)


def write_avi(path: Path, frames):
    largest = max(map(len, frames))
    avih = struct.pack("<14I", round(1_000_000 / FPS), largest * FPS, 0, 0x10, len(frames), 0, 1, largest, WIDTH, HEIGHT, 0, 0, 0, 0)
    strh = struct.pack("<4s4sIHH8I4h", b"vids", b"MJPG", 0, 0, 0, 0, 1, FPS, 0, len(frames), largest, 0xFFFFFFFF, 0, 0, 0, WIDTH, HEIGHT)
    strf = struct.pack("<IiiHH4sIiiII", 40, WIDTH, HEIGHT, 1, 24, b"MJPG", WIDTH * HEIGHT * 3, 0, 0, 0, 0)
    header = list_chunk(b"hdrl", chunk(b"avih", avih) + list_chunk(b"strl", chunk(b"strh", strh) + chunk(b"strf", strf)))
    movi, index, offset = bytearray(), bytearray(), 4
    for frame in frames:
        item = chunk(b"00dc", frame)
        movi.extend(item)
        index.extend(struct.pack("<4sIII", b"00dc", 0x10, offset, len(frame)))
        offset += len(item)
    body = header + list_chunk(b"movi", bytes(movi)) + chunk(b"idx1", bytes(index))
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body) + 4) + b"AVI " + body)


def encode(image):
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=91, subsampling=0, optimize=True)
    return buffer.getvalue()


def main():
    root = Path(__file__).resolve().parent
    run_dir = root / "run" / "post-collapse-debug"
    output_dir = root / "output" / "postcollapse_movie"
    output_dir.mkdir(parents=True, exist_ok=True)

    simulation = read_half_frames(run_dir / "WALL-CAVITATION-HALF.plt")
    ratios = read_history(run_dir / "RATIO.TXT")
    centroids = read_history(run_dir / "CENT.TXT")
    postcollapse = read_postcollapse(run_dir / "POSTCOLLAPSE_DIAG.dat")
    if min(len(ratios), len(centroids)) < len(simulation) + 1:
        raise ValueError("History files do not contain one record per simulation frame")
    if not postcollapse:
        raise ValueError("POSTCOLLAPSE_DIAG.dat contains no post-collapse records")

    times = [ratios[index][0] for index in range(1, len(simulation) + 1)]
    post_start = min(range(len(times)), key=lambda index: abs(times[index] - postcollapse[0][0]))
    if abs(times[post_start] - postcollapse[0][0]) > 5e-6:
        raise ValueError("Could not align post-collapse diagnostics with geometry")
    if len(simulation) - post_start != len(postcollapse):
        raise ValueError(f"Post-collapse geometry/diagnostic count mismatch: {len(simulation) - post_start} versus {len(postcollapse)}")
    for offset, row in enumerate(postcollapse):
        if abs(times[post_start + offset] - row[0]) > 5e-6:
            raise ValueError(f"Post-collapse time mismatch at diagnostic row {offset + 1}")

    encoded = []
    preview = None
    for index, (positive, negative) in enumerate(simulation):
        history_index = index + 1
        time_value, ratio = ratios[history_index]
        _, centroid = centroids[history_index]
        ring = None
        if index >= post_start:
            diagnostic = postcollapse[index - post_start]
            ring = (diagnostic[2], diagnostic[3])
        image = render(positive, negative, history_index, len(simulation), time_value, ratio, centroid, ring)
        encoded.append(encode(image))
        if index == post_start:
            preview = image.copy()

    # Every state appears once; brief holds make the transition readable.
    timeline = [encoded[0]] * FPS
    for index, frame in enumerate(encoded):
        timeline.append(frame)
        if index in (post_start - 1, post_start):
            timeline.extend([frame] * (FPS // 2))
    timeline.extend([encoded[-1]] * FPS)

    avi_path = output_dir / "postcollapse_movie.avi"
    preview_path = output_dir / "postcollapse_movie-preview.png"
    write_avi(avi_path, timeline)
    preview.save(preview_path)
    print(f"Wrote {len(timeline)} AVI frames at {FPS} fps")
    print(f"Included all {len(simulation)} accepted simulation steps")
    print(f"Post-collapse: {len(postcollapse)} steps, simulation frames {post_start + 1}-{len(simulation)}")
    print(f"AVI: {avi_path}")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()
