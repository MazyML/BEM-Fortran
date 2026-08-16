#!/usr/bin/env python3
"""Render NOZZLE1.plt from the BEM run as a Motion-JPEG AVI."""

from __future__ import annotations

import io
import math
import re
import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 15
PLOT = (80, 92, 1200, 652)
XLIM, YLIM = (-8.4, 8.4), (-5.2, 3.2)


def zones(path: Path):
    lines = path.read_text().splitlines()
    result, i = [], 0
    while i < len(lines):
        match = re.search(r"ZONE\s+I=\s*(\d+)", lines[i])
        if not match:
            i += 1
            continue
        count = int(match.group(1))
        points = [tuple(map(float, row.split()[:4])) for row in lines[i + 1 : i + count + 1]]
        if len(points) != count:
            raise ValueError(f"Incomplete zone near line {i + 1}")
        result.append(points)
        i += count + 1
    if len(result) % 2:
        raise ValueError("Each frame must have a bubble zone and a wall zone")
    return list(zip(result[0::2], result[1::2]))


def history(path: Path):
    return [tuple(map(float, line.split()[:2])) for line in path.read_text().splitlines() if len(line.split()) >= 2]


def font(size: int, bold=False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    try:
        return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)
    except OSError:
        return ImageFont.load_default()


def point_xy(point, scale=2):
    left, top, right, bottom = (v * scale for v in PLOT)
    x, y = point[:2]
    px = left + (x - XLIM[0]) / (XLIM[1] - XLIM[0]) * (right - left)
    py = bottom - (y - YLIM[0]) / (YLIM[1] - YLIM[0]) * (bottom - top)
    return round(px), round(py)


def wall_sides(points):
    gaps = [math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1]) for i in range(len(points) - 1)]
    cut = max(range(len(gaps)), key=gaps.__getitem__) + 1
    return points[:cut], points[cut:]


def vertical_label(image, text, body_font, scale=2):
    probe = ImageDraw.Draw(image)
    box = probe.textbbox((0, 0), text, font=body_font)
    label = Image.new("RGBA", (box[2] + 8 * scale, box[3] + 8 * scale), (0, 0, 0, 0))
    ImageDraw.Draw(label).text((4 * scale, 4 * scale), text, font=body_font, fill=(160, 181, 194))
    label = label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(label, (19 * scale, (H * scale - label.height) // 2), label)


def render(bubble, wall, step, total, time_value, ratio):
    scale = 2
    image = Image.new("RGB", (W * scale, H * scale), (7, 16, 29))
    draw = ImageDraw.Draw(image)
    title, body, small = font(29 * scale, True), font(18 * scale), font(15 * scale)

    draw.text((80 * scale, 28 * scale), "Cavitational bubble collapse near a bilge opening", font=title, fill=(235, 245, 250))
    draw.text((1200 * scale, 38 * scale), "Axisymmetric BEM", font=body, fill=(126, 156, 178), anchor="ra")

    left, top, right, bottom = (v * scale for v in PLOT)
    draw.rounded_rectangle((left, top, right, bottom), radius=12 * scale, fill=(11, 27, 43), outline=(47, 73, 92), width=2 * scale)
    for tick in range(-8, 9, 2):
        a, b = point_xy((tick, YLIM[0]), scale), point_xy((tick, YLIM[1]), scale)
        draw.line((a, b), fill=(26, 48, 65), width=scale)
        draw.text((a[0], bottom + 11 * scale), str(tick), font=small, fill=(121, 151, 170), anchor="ma")
    for tick in range(-5, 4):
        a, b = point_xy((XLIM[0], tick), scale), point_xy((XLIM[1], tick), scale)
        draw.line((a, b), fill=(26, 48, 65), width=scale)
        draw.text((left - 12 * scale, a[1]), str(tick), font=small, fill=(121, 151, 170), anchor="rm")
    draw.line((point_xy((0, YLIM[0]), scale), point_xy((0, YLIM[1]), scale)), fill=(78, 109, 129), width=scale)

    for side in wall_sides(wall):
        polygon = [point_xy(p, scale) for p in side]
        draw.polygon(polygon, fill=(74, 83, 91), outline=(170, 183, 192))
        draw.line(polygon + [polygon[0]], fill=(178, 190, 198), width=2 * scale, joint="curve")
    bubble_polygon = [point_xy(p, scale) for p in bubble]
    draw.polygon(bubble_polygon, fill=(12, 158, 174))
    draw.line(bubble_polygon + [bubble_polygon[0]], fill=(151, 244, 244), width=3 * scale, joint="curve")

    draw.text((98 * scale, 112 * scale), f"step {step:03d} / {total}", font=body, fill=(221, 235, 242))
    draw.text((98 * scale, 141 * scale), f"t* = {time_value:.5f}", font=body, fill=(148, 187, 207))
    draw.text((98 * scale, 170 * scale), f"V / V0 = {ratio:.5f}", font=body, fill=(116, 225, 225))
    draw.text((640 * scale, 687 * scale), "radial coordinate r / R0", font=body, fill=(160, 181, 194), anchor="ma")
    vertical_label(image, "axial coordinate z / R0", body, scale)

    x0, y0, x1, y1 = 930 * scale, 116 * scale, 1167 * scale, 132 * scale
    draw.rounded_rectangle((x0, y0, x1, y1), radius=8 * scale, fill=(28, 52, 69))
    draw.rounded_rectangle((x0, y0, x0 + (x1 - x0) * step / total, y1), radius=8 * scale, fill=(21, 190, 198))
    draw.text((x1, 142 * scale), "collapse progress", font=small, fill=(127, 159, 176), anchor="ra")
    return image.resize((W, H), Image.Resampling.LANCZOS)


def chunk(tag: bytes, payload: bytes):
    return tag + struct.pack("<I", len(payload)) + payload + (b"\0" if len(payload) & 1 else b"")


def list_chunk(kind: bytes, payload: bytes):
    return chunk(b"LIST", kind + payload)


def write_avi(path: Path, frames):
    largest = max(map(len, frames))
    avih = struct.pack("<14I", round(1_000_000 / FPS), largest * FPS, 0, 0x10, len(frames), 0, 1, largest, W, H, 0, 0, 0, 0)
    strh = struct.pack("<4s4sIHH8I4h", b"vids", b"MJPG", 0, 0, 0, 0, 1, FPS, 0, len(frames), largest, 0xFFFFFFFF, 0, 0, 0, W, H)
    strf = struct.pack("<IiiHH4sIiiII", 40, W, H, 1, 24, b"MJPG", W * H * 3, 0, 0, 0, 0)
    hdrl = list_chunk(b"hdrl", chunk(b"avih", avih) + list_chunk(b"strl", chunk(b"strh", strh) + chunk(b"strf", strf)))

    movi, index, offset = bytearray(), bytearray(), 4
    for frame in frames:
        item = chunk(b"00dc", frame)
        movi.extend(item)
        index.extend(struct.pack("<4sIII", b"00dc", 0x10, offset, len(frame)))
        offset += len(item)
    body = hdrl + list_chunk(b"movi", bytes(movi)) + chunk(b"idx1", bytes(index))
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body) + 4) + b"AVI " + body)


def jpeg(image):
    output = io.BytesIO()
    image.save(output, "JPEG", quality=91, subsampling=0, optimize=True)
    return output.getvalue()


def main():
    here = Path(__file__).resolve().parent
    simulation = zones(here / "NOZZLE1.plt")
    values = history(here / "RATIO.TXT")
    if len(values) < len(simulation) + 1:
        raise ValueError("RATIO.TXT has fewer values than NOZZLE1.plt has frames")

    encoded, middle = [], None
    for index, (bubble, wall) in enumerate(simulation, 1):
        time_value, ratio = values[index]
        image = render(bubble, wall, index, len(simulation), time_value, ratio)
        encoded.append(jpeg(image))
        if index == len(simulation) // 2:
            middle = image.copy()
    encoded = [encoded[0]] * FPS + encoded + [encoded[-1]] * FPS
    write_avi(here / "bilge-collapse.avi", encoded)
    middle.save(here / "bilge-collapse-preview.png")
    print(f"Wrote {len(encoded)} AVI frames ({len(simulation)} simulation frames) at {FPS} fps")


if __name__ == "__main__":
    main()
