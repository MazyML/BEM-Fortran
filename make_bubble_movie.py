#!/usr/bin/env python3
"""Plain movie of the bubble surface, straight from WALL-CAVITATION.plt.

One frame per accepted step, the bubble filled on a white ground with the
rigid wall as a band along the bottom, and nothing on it but the frame
number and the physical time.  A continuation run is a chain of restarts,
so several run directories are given and stitched in the order listed:

    python make_bubble_movie.py bubble.avi run/parent run/chain/gen*

There is no ffmpeg here, so the AVI is written by the MJPEG writer in
make_wall_movie.py, which lives beside this script.

Two things about the .plt are worth knowing before changing any of this;
both were found the hard way and both are commented where they are handled.
The solver writes a single mirrored contour per frame, which is one curve
while the bubble is simply connected and two disjoint rings once it is
toroidal - see split_axis_loops.  And the frames have no timestamps, so
they have to be paired with a diagnostic that has one row per accepted
step - see load_times.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# The MJPEG writer, the JPEG encoder and the font loader all live in the
# older movie script next to this one.
_spec = importlib.util.spec_from_file_location(
    "make_wall_movie", Path(__file__).resolve().parent / "make_wall_movie.py")
_mwm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mwm)

# White ground, gas a pale fill with a dark edge, wall a grey band.
LIQUID = (255, 255, 255)
GAS = (219, 232, 244)
EDGE = (23, 42, 63)
WALL = (206, 206, 206)
WALL_LINE = (60, 60, 60)
TEXT = (60, 60, 60)


def read_zones(path):
    """Every ZONE in a .plt as an array of (r, z).

    The file is written with FORMAT(4F10.4), so the fields are ten columns
    wide.  The velocity columns can overflow that and run into each other,
    which leaves a line that cannot be split on whitespace, so the two
    columns that are wanted are sliced by position instead.
    """
    lines = Path(path).read_text().splitlines()
    zones, i = [], 0
    while i < len(lines):
        match = re.search(r"ZONE\s+I=\s*(\d+)", lines[i])
        if not match:
            i += 1
            continue
        count = int(match.group(1))
        points = []
        for row in lines[i + 1:i + 1 + count]:
            try:
                points.append((float(row[0:10]), float(row[10:20])))
            except ValueError:
                parts = row.split()
                if len(parts) >= 2:
                    try:
                        points.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
        zones.append(np.array(points))
        i += count + 1
    return zones


def read_frames(path):
    """Group the zone stream into frames.

    Each accepted step writes the bubble - one zone while it is a single
    loop, one per half per component once it has split - and then the wall.
    The wall is the zone that lies entirely on z = 0, and it closes a frame.
    """
    frames, current = [], []
    for zone in read_zones(path):
        if len(zone) and np.max(np.abs(zone[:, 1])) < 1e-9:
            if current:
                frames.append(current)
            current = []
        else:
            current.append(zone)
    if current:
        frames.append(current)
    return frames


def split_axis_loops(zone):
    """A toroidal frame arrives as one path, not one curve.

    The solver mirrors its contour for output as GSC(1..M+1) = the contour
    and GSC(M+2..2M+1) = its reflection, and writes the whole thing as a
    single zone.  While the bubble is simply connected that really is one
    curve: it starts and ends on the axis, so the path closes with nothing
    spurious in it.  Once the bubble is toroidal the contour closes on
    itself away from the axis, so the same list holds two disjoint rings
    with the joins between them running straight across the hole - filled
    as one polygon they seal the hole shut and put a wedge on the
    interface.

    The tell is GSC(M+1) = GSC(1): the first half has already closed.
    """
    n = len(zone)
    if n % 2 == 0 or n < 5:
        return [zone]
    half = (n - 1) // 2
    if np.hypot(*(zone[0] - zone[half])) > 1e-9:
        return [zone]
    return [zone[:half + 1], zone[half + 1:]]


def _energy_trace_times(directory):
    """(time, volume) per accepted step from ENERGY_TRACE.dat, in step order.

    The trace is written on the ILS=1 solve, and a relief, a
    redistribution, an absorption or the energy jump all re-solve at the
    same step, so a step can appear more than once.  The .plt writes
    exactly one frame per step, at ILS=2.  Keying on the step number and
    keeping the last row for each is what lines the two up.
    """
    path = os.path.join(directory, "ENERGY_TRACE.dat")
    if not os.path.exists(path):
        return []
    seen, order = {}, []
    for line in open(path):
        if line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) > 4 and parts[0].isdigit() and parts[1].isdigit():
            try:
                key = (int(parts[0]), int(parts[1]))
            except ValueError:
                continue
            if key not in seen:
                order.append(key)
            try:
                seen[key] = float(parts[2])
            except ValueError:
                order.pop()
    return [seen[k] for k in order]


def _ratio_txt_times(directory):
    """Times from RATIO.TXT, which is written in every phase, one row a step.

    Only five decimals, so it is the fallback rather than the first choice.
    """
    path = os.path.join(directory, "RATIO.TXT")
    if not os.path.exists(path):
        return []
    times = []
    for line in open(path):
        parts = line.split()
        if len(parts) == 2:
            try:
                times.append(float(parts[0]))
            except ValueError:
                pass
    return times[1:]


def load_times(directory, nframes):
    """Pick whichever source covers all the frames in this directory.

    ENERGY_TRACE is the better source, but a run made with a build whose
    trace still stopped at the jet impact has no rows for its toroidal
    frames; RATIO.TXT is written in every phase and covers those.
    """
    times = _energy_trace_times(directory)
    if len(times) >= nframes:
        return times
    fallback = _ratio_txt_times(directory)
    return fallback if len(fallback) > len(times) else times


def collect(directories):
    """(time, zones) for every frame, in the order the directories are given."""
    sequence = []
    for directory in directories:
        plt_path = os.path.join(directory, "WALL-CAVITATION.plt")
        if not os.path.exists(plt_path):
            print(f"  {directory}: no WALL-CAVITATION.plt, skipped")
            continue
        frames = read_frames(plt_path)
        times = load_times(directory, len(frames))
        if not frames or not times:
            print(f"  {directory}: no frames or no times, skipped")
            continue
        n = min(len(frames), len(times))
        if abs(len(frames) - len(times)) > 2:
            print(f"  {directory}: {len(frames)} frames against {len(times)} "
                  f"step times - pairing the first {n}")
        sequence.extend((times[k], frames[k]) for k in range(n))
        print(f"  {directory}: {n} frames, "
              f"t {times[0]:.6f}..{times[n - 1]:.6f}")
    return sequence


def render(sequence, width, height, fps, out_path):
    zones = [z for _, frame in sequence for z in frame]
    xs = np.concatenate([z[:, 0] for z in zones])
    ys = np.concatenate([z[:, 1] for z in zones])

    # One fixed view for the whole movie, so sizes can be compared by eye.
    ztop = ys.max() * 1.06
    xhalf = max(abs(xs).max() * 1.04, ztop * width / height / 2)
    scale = min(width / (2 * xhalf), height / (1.06 * ztop))
    midx, midy = 0.0, 0.5 * (-0.06 * ztop + ztop)
    cx, cy = width / 2, height / 2

    def to_pixels(points):
        return [(cx + (a - midx) * scale, cy - (b - midy) * scale)
                for a, b in points]

    wall_y = cy + midy * scale
    font = _mwm.load_font(22)

    encoded = []
    for i, (time_value, frame) in enumerate(sequence):
        image = Image.new("RGB", (width, height), LIQUID)
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, wall_y, width, height], fill=WALL)
        for zone in frame:
            for piece in split_axis_loops(zone):
                if len(piece) >= 3:
                    pixels = to_pixels(piece)
                    draw.polygon(pixels, fill=GAS)
                    draw.line(pixels + [pixels[0]], fill=EDGE, width=2)
        draw.line([(0, wall_y), (width, wall_y)], fill=WALL_LINE, width=3)
        draw.text((16, 12), f"frame {i + 1}", font=font, fill=TEXT)
        draw.text((16, 40), f"t = {time_value:.6f}", font=font, fill=TEXT)
        encoded.append(_mwm.encode(image))
        if i % 500 == 0:
            print(f"  {i}/{len(sequence)}")

    # write_avi reads these from its own module scope.
    _mwm.WIDTH, _mwm.HEIGHT, _mwm.FPS = width, height, fps
    _mwm.write_avi(Path(out_path), encoded)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", help="AVI to write")
    parser.add_argument("runs", nargs="+",
                        help="run directories, in order; globs are expanded")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=820)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--every", type=int, default=1, metavar="N",
                        help="keep every Nth frame (default 1, all of them)")
    parser.add_argument("--daughter-every", type=int, default=1, metavar="N",
                        help="keep every Nth frame while the bubble is split; "
                             "a frame carrying more than one zone is one of "
                             "those, since the solver writes a zone per half "
                             "per component once it has split")
    args = parser.parse_args(argv)

    directories = []
    for pattern in args.runs:
        matched = sorted(glob.glob(pattern))
        directories.extend(matched if matched else [pattern])

    print("reading:")
    sequence = collect(directories)
    if not sequence:
        print("no frames found")
        return 1

    kept, ndaughter = [], 0
    for k, item in enumerate(sequence):
        if len(item[1]) > 1:
            ndaughter += 1
            if ndaughter % args.daughter_every:
                continue
        elif k % args.every:
            continue
        kept.append(item)

    times = [t for t, _ in kept]
    print(f"frames: {len(kept)} of {len(sequence)}   "
          f"t {min(times):.6f}..{max(times):.6f}")
    backward = sum(1 for i in range(1, len(times)) if times[i] < times[i - 1])
    if backward:
        print(f"WARNING: {backward} frames go backwards in time - the frame "
              f"to step pairing is wrong for at least one directory")

    render(kept, args.width, args.height, args.fps, args.output)
    size = os.path.getsize(args.output) / 1e6
    print(f"wrote {args.output}  {size:.1f} MB, {len(kept)} frames "
          f"@ {args.fps} fps, {len(kept) / args.fps:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
