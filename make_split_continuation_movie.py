#!/usr/bin/env python3
"""Render the post-split continuation of the flat-wall BEM run as an MP4.

The continuation is chained.  Each generation restarts from the full-precision
pre-contact checkpoint written by the previous one, splits the contour at that
contact, discards the smaller daughter at the moment it is created, and carries
the retained daughter on to its next contact.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent / ".tools" / "python-packages"))
import imageio_ffmpeg  # noqa: E402

ROOT = Path(__file__).resolve().parent
GENERATIONS = [ROOT / "run" / "final_abs" / ("gen%d" % n) for n in range(1, 13)]
OUTPUT = ROOT / "output" / "split-continuation"

# Generations 3 and 4 spend thousands of steps advancing the clock very little,
# so they are sampled evenly in physical time instead of frame by frame.
# Generations 1 and 11 cover most of the elapsed time; the nine between them
# are short hops from one coalescence to the next, so they get a smaller share.
SAMPLE_TARGET = dict([(1, 110), (2, 60), (11, 110), (12, 200)]
                     + [(n, 48) for n in range(3, 11)])

# A generation is truncated at the last state whose peak speed is still
# physical.  The solver now stops itself on the time-step floor, so this rarely
# bites: healthy generations peak between 450 and 1800 near their contacts;
# generation 3 crosses it as the surface velocity runs away and the time step
# collapses to round-off, and nothing past that point carries physical content.
# With the contact captured while the neck is still resolved, no generation now
# approaches this bound - peak speed stays between 137 and 222 throughout - so
# nothing is truncated or flagged.  The check is kept as a guard: a generation
# already above the bound on its first state would be marked rather than cut,
# since truncating it would leave nothing to show.
MAXU_LIMIT = 1.0e4

# A rebound cannot pass the bubble it started from, so anything approaching
# V/V0 = 1 is not physical.  Past VOLUME_TRUST the expansion is being driven by
# a surface potential the surgeries have inflated - it reaches V/V0 = 12 if left
# alone - so those states are banded, and the run is cut at VOLUME_LIMIT rather
# than showing a balloon that means nothing.
VOLUME_TRUST = 0.28
VOLUME_LIMIT = 0.45

# Beyond this share of the parent circulation, discarding a daughter is no
# longer a small correction and the generation is flagged on screen.
CIRCULATION_WARN = 1.0

WIDTH, HEIGHT, FPS, SS = 1600, 896, 30, 2
MAIN = (120, 118, 1020, 782)
INSET = (1100, 140, 1520, 520)
VIEW_Z = (0.25, 1.98)
_aspect = (MAIN[2] - MAIN[0]) / (MAIN[3] - MAIN[1])
_rspan = (VIEW_Z[1] - VIEW_Z[0]) * _aspect
VIEW_R = (-_rspan / 2, _rspan / 2)
INSET_R, INSET_Z = (-1.45, 1.45), (-0.1, 2.8)

V0 = 4.1887902047863905    # 4/3 pi: the initial bubble volume in R0 units

BG = (7, 16, 29)
PANEL = (11, 27, 43)
GRID = (26, 48, 65)
EDGE = (47, 73, 92)
INK = (235, 245, 250)
DIM = (126, 156, 178)
FAINT = (105, 132, 150)
BUBBLE = (12, 158, 174)
BUBBLE_EDGE = (151, 244, 244)
SMALLER = (196, 118, 42)
SMALLER_EDGE = (255, 196, 118)
RING = (244, 117, 204)
ACCENT = (255, 190, 77)
WARN = (255, 128, 110)


# --------------------------------------------------------------------- inputs
def read_zones(path):
    """Zones are written as 4F10.4.  The velocity columns overflow and run
    together once the jet is fast, so coordinates are taken by column."""
    lines = path.read_text().splitlines()
    zones, index = [], 0
    while index < len(lines):
        match = re.search(r"ZONE\s+I=\s*(\d+)", lines[index])
        if not match:
            index += 1
            continue
        count = int(match.group(1))
        rows = lines[index + 1 : index + 1 + count]
        if len(rows) != count:
            raise ValueError("incomplete zone near line %d of %s" % (index + 1, path))
        zones.append([(float(row[0:10]), float(row[10:20])) for row in rows])
        index += count + 1
    return zones


def read_frames(path):
    """A frame is every bubble contour up to the mirrored pair of wall zones."""
    zones = read_zones(path)
    frames, current, index = [], [], 0
    while index < len(zones):
        if all(abs(point[1]) < 1e-9 for point in zones[index]):
            frames.append(current)
            current = []
            index += 2
        else:
            current.append(zones[index])
            index += 1
    return frames


def read_diag(path):
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] == "#":
            continue
        rows.append([float(value) for value in fields])
    return rows


def read_log(path):
    """Circulation bookkeeping and the ending of one generation."""
    text = path.read_text()
    info = {"parent": 0.0, "dropped": 0.0, "dropped_v": 0.0, "dropped_fraction": 0.0,
            "kept_both": False, "degenerate": False, "ending": "run ended"}
    match = re.search(r"SPLIT CIRCULATIONS FROM CHECKPOINT: PARENT=\s*(\S+)", text)
    if match:
        info["parent"] = float(match.group(1))
    match = re.search(r"SATELLITE DISCARDED AT CREATION: VOLUME=\s*(\S+)\s+"
                      r"FRACTION=\s*(\S+)\s+CIRCULATION=\s*(\S+)", text)
    if match:
        info["dropped_v"] = float(match.group(1))
        info["dropped_fraction"] = float(match.group(2))
        info["dropped"] = float(match.group(3))
    info["kept_both"] = "BOTH DAUGHTERS KEPT" in text
    if "FULL-PRECISION PRECONTACT STATE SAVED" in text:
        info["ending"] = "contact captured"
    elif "NON-POSITIVE ACCEPTED TIME" in text or "TIME STEP BELOW FLOOR" in text:
        info["ending"] = "time step underflow"
    elif "UNRESOLVED THIRD TOPOLOGY" in text:
        info["ending"] = "unresolved reconnection"
    if "R(J)<0.0001" in text:
        info["ending"] = "surface reached the axis"
    return info


def read_absorptions(path):
    """Points the closure took in after the bubble was made whole."""
    events = {}
    if not path.exists():
        return events
    for line in path.read_text().splitlines():
        f = line.split()
        if not f or f[0].startswith("#"):
            continue
        row = [float(v) for v in f]
        events[int(row[2])] = dict(radius=row[5], dropped=row[9], fraction=row[10])
    return events


def read_axis(path):
    """The step at which the torus was made whole again, if it happened."""
    if not path.exists():
        return None
    info = {}
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) < 2 or f[0] != "#":
            continue
        if f[1] == "TIME" and len(f) >= 6:
            info["post"] = int(float(f[5]))
        elif f[1] == "RELEASED_CIRCULATION" and len(f) >= 3:
            info["gamma"] = float(f[2])
        elif f[1] == "RMIN" and len(f) >= 7:
            info["rmin"] = float(f[4])
        elif f[1] == "VOLUME_BEFORE" and len(f) >= 7:
            info["dv"] = float(f[6])
    return info if "post" in info else None


def read_merges(path):
    """Every coalescence in one generation, keyed by the post-step it happened."""
    events = {}
    if not path.exists():
        return events
    current = {}
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) < 2 or f[0] != "#":
            continue
        tag = f[1]
        if tag == "MERGE":
            current = {}
        elif tag == "STEP" and len(f) >= 7:
            current["post"] = int(float(f[5]))
        elif tag == "GAP" and len(f) >= 7:
            current["ratio"] = float(f[6])
        elif tag == "GAMMA1" and len(f) >= 10:
            current["sum"] = float(f[8])
            current["merged"] = float(f[9])
        elif tag == "VOLUME_BEFORE" and len(f) >= 7:
            current["dv"] = float(f[6])
            if "post" in current:
                events[current["post"]] = dict(current)
    return events


def read_retirements(path):
    """Every component removal in one generation, keyed by post-step."""
    events = {}
    if not path.exists():
        return events
    for line in path.read_text().splitlines():
        f = line.split()
        if not f or f[0] == "#":
            continue
        row = [float(v) for v in f]
        events[int(row[2])] = dict(volume=row[6], fraction=row[7], gamma=row[9],
                                   ratio=row[12])
    return events


def load_generation(directory):
    frames = read_frames(directory / "WALL-CAVITATION-HALF.plt")
    diag = read_diag(directory / "MULTICOMP_DIAG.dat")
    # A generation that dies mid-step never reaches the plot write for its last
    # accepted state, so the two lists are trimmed to their common length.
    available = min(len(frames), len(diag))
    log = read_log(directory / "out.txt")
    count = available
    if diag[0][16] > MAXU_LIMIT:
        # Already past the bound on its first state, so truncating would leave
        # nothing.  It is kept whole and marked instead.
        log["degenerate"] = True
    else:
        for index in range(available):
            if diag[index][16] > MAXU_LIMIT:
                count = index
                break
        if count < available:
            log["ending"] = "peak speed runs away"
    for index in range(count):
        if diag[index][6] > VOLUME_LIMIT:
            count = index
            log["ending"] = "expansion over-driven"
            break
    log["retirements"] = read_retirements(directory / "COMPONENT_RETIREMENT_DIAG.dat")
    log["merges"] = read_merges(directory / "MERGE_DIAG.dat")
    log["absorb"] = read_absorptions(directory / "AXIS_ABSORB_DIAG.dat")
    log["axis"] = read_axis(directory / "AXIS_RECONNECTION.dat")
    return frames[:count], diag[:count], log


def sample_evenly_in_time(diag, target):
    """Pick frames for one generation.

    Half the budget is spread evenly in physical time, so a phase that covers a
    lot of clock is not compressed away.  The other half is spread evenly in
    step index, because a phase where the clock crawls collapses to almost
    nothing under time sampling alone even though the shape is still moving.
    """
    count = len(diag)
    if count <= target:
        return list(range(count))
    times = [row[2] for row in diag]
    first, last = times[0], times[-1]
    picks, cursor = {0, count - 1}, 0
    budget = max(2, target // 2)
    for step in range(budget):
        goal = first + (last - first) * step / (budget - 1)
        while cursor + 1 < count and times[cursor + 1] <= goal:
            cursor += 1
        picks.add(cursor)
    stride = max(1, count // max(1, target - budget))
    picks.update(range(0, count, stride))
    return sorted(picks)


# --------------------------------------------------------------------- canvas
def font(size, bold=False):
    names = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def mapper(box, xlim, ylim):
    left, top, right, bottom = (value * SS for value in box)

    def to_pixels(point):
        x, y = point[0], point[1]
        return (
            round(left + (x - xlim[0]) / (xlim[1] - xlim[0]) * (right - left)),
            round(bottom - (y - ylim[0]) / (ylim[1] - ylim[0]) * (bottom - top)),
        )

    return to_pixels


def contour(draw, zones, to_pixels, fill, edge, width=3):
    """A toroidal component is a closed loop clear of the axis, so its two
    signed-r contours are separate curves and are filled separately; merging
    them would paint a bridge across r = 0 that is not part of the bubble.

    A component that has been made whole again is the other case: an open arc
    with both ends on the axis.  There the two halves are one surface, so they
    are joined into a single polygon.  Filling them separately would close each
    half along r = 0 and draw a seam up the middle of a bubble that has no
    boundary there at all."""
    positive = zones[0]
    reaches_axis = (len(zones) == 2
                    and abs(positive[0][0]) < 1.0e-6
                    and abs(positive[-1][0]) < 1.0e-6)
    if reaches_axis:
        polygon = ([to_pixels(point) for point in positive]
                   + [to_pixels(point) for point in reversed(zones[1])])
        draw.polygon(polygon, fill=fill)
        draw.line(polygon + [polygon[0]], fill=edge, width=width * SS, joint="curve")
        return
    for zone in zones:
        polygon = [to_pixels(point) for point in zone]
        draw.polygon(polygon, fill=fill)
        draw.line(polygon + [polygon[0]], fill=edge, width=width * SS, joint="curve")


def ticks(draw, box, xlim, ylim, to_pixels, xvals, yvals, small):
    left, _top, _right, bottom = (value * SS for value in box)
    for value in xvals:
        a = to_pixels((value, ylim[0]))
        draw.line((a, to_pixels((value, ylim[1]))), fill=GRID, width=SS)
        draw.text((a[0], bottom + 10 * SS), "%g" % value, font=small, fill=FAINT, anchor="ma")
    for value in yvals:
        a = to_pixels((xlim[0], value))
        draw.line((a, to_pixels((xlim[1], value))), fill=GRID, width=SS)
        draw.text((left - 12 * SS, a[1]), "%g" % value, font=small, fill=FAINT, anchor="rm")


def vertical_label(image, text, text_font, x, colour):
    box = ImageDraw.Draw(image).textbbox((0, 0), text, font=text_font)
    label = Image.new("RGBA", (box[2] + 8 * SS, box[3] + 8 * SS), (0, 0, 0, 0))
    ImageDraw.Draw(label).text((4 * SS, 4 * SS), text, font=text_font, fill=colour)
    label = label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(label, (x * SS, (HEIGHT * SS - label.height) // 2), label)


def render(state, index, count):
    image = Image.new("RGB", (WIDTH * SS, HEIGHT * SS), BG)
    draw = ImageDraw.Draw(image)
    title = font(30 * SS, True)
    body = font(19 * SS)
    small = font(15 * SS)
    mono = font(20 * SS, True)

    draw.text((120 * SS, 30 * SS),
              "Cavitation bubble above a rigid wall: continuation after the split at contact",
              font=title, fill=INK)
    draw.text((120 * SS, 72 * SS),
              "Axisymmetric BEM  -  gamma = 1.5  -  daughters merge when they touch; a torus whose hole closes is made whole",
              font=body, fill=DIM)

    # ---- main zoomed view
    left, top, right, bottom = (value * SS for value in MAIN)
    draw.rounded_rectangle((left, top, right, bottom), radius=12 * SS,
                           fill=PANEL, outline=EDGE, width=2 * SS)
    to_main = mapper(MAIN, VIEW_R, VIEW_Z)
    ticks(draw, MAIN, VIEW_R, VIEW_Z, to_main,
          (-1.0, -0.5, 0.0, 0.5, 1.0),
          (0.4, 0.8, 1.2, 1.6), small)
    draw.line((to_main((0.0, VIEW_Z[0])), to_main((0.0, VIEW_Z[1]))),
              fill=(78, 109, 129), width=SS)

    # Components are coloured by the volume they enclose, not by their slot: the
    # surgery names the loops from their arc lengths, and the loop it calls
    # dominant is not always the larger one.
    for zones, volume, _ring in state["components"]:
        larger = volume >= 0.5 * state["volume"]
        contour(draw, zones, to_main, BUBBLE if larger else SMALLER,
                BUBBLE_EDGE if larger else SMALLER_EDGE, 3 if larger else 2)
    for _zones, _volume, ring in state["components"]:
        for signed in (-ring[0], ring[0]):
            cx, cy = to_main((signed, ring[1]))
            marker = 5 * SS
            draw.ellipse((cx - marker, cy - marker, cx + marker, cy + marker),
                         fill=RING, outline=(255, 218, 244), width=SS)

    draw.text(((left + right) // 2, bottom + 32 * SS), "radial coordinate  r / R0",
              font=body, fill=DIM, anchor="ma")
    vertical_label(image, "height above wall  z / R0", body, 24, DIM)

    # ---- context inset
    ileft, itop, iright, ibottom = (value * SS for value in INSET)
    draw.rounded_rectangle((ileft, itop, iright, ibottom), radius=10 * SS,
                           fill=PANEL, outline=EDGE, width=2 * SS)
    to_inset = mapper(INSET, INSET_R, INSET_Z)
    wall_y = to_inset((0.0, 0.0))[1]
    draw.rectangle((ileft, wall_y, iright, ibottom), fill=(52, 58, 65))
    draw.line((ileft, wall_y, iright, wall_y), fill=(207, 218, 224), width=3 * SS)
    upper, lower = to_inset((-1.0, 2.5)), to_inset((1.0, 0.5))
    draw.ellipse((upper[0], upper[1], lower[0], lower[1]), outline=(46, 74, 94), width=2 * SS)
    draw.text((iright - 10 * SS, itop + 8 * SS), "initial bubble",
              font=small, fill=(74, 104, 124), anchor="ra")
    corner_a, corner_b = to_inset((VIEW_R[0], VIEW_Z[1])), to_inset((VIEW_R[1], VIEW_Z[0]))
    draw.rectangle((corner_a[0], corner_a[1], corner_b[0], corner_b[1]),
                   outline=ACCENT, width=2 * SS)
    draw.text((corner_b[0] + 8 * SS, (corner_a[1] + corner_b[1]) // 2), "view",
              font=small, fill=ACCENT, anchor="lm")
    draw.text((ileft + 12 * SS, wall_y - 24 * SS), "rigid wall  -  z = 0",
              font=small, fill=(198, 210, 218))

    # ---- legend
    x = 1100 * SS
    legend_y = 544 * SS
    draw.rectangle((x, legend_y, x + 24 * SS, legend_y + 14 * SS),
                   fill=BUBBLE, outline=BUBBLE_EDGE, width=SS)
    draw.text((x + 34 * SS, legend_y - 3 * SS),
              "larger daughter" if len(state["components"]) > 1 else "bubble",
              font=small, fill=DIM)
    if len(state["components"]) > 1:
        draw.rectangle((x + 200 * SS, legend_y, x + 224 * SS, legend_y + 14 * SS),
                       fill=SMALLER, outline=SMALLER_EDGE, width=SS)
        draw.text((x + 234 * SS, legend_y - 3 * SS), "smaller daughter", font=small, fill=DIM)
    draw.ellipse((x + 6 * SS, legend_y + 26 * SS, x + 18 * SS, legend_y + 38 * SS),
                 fill=RING, outline=(255, 218, 244), width=SS)
    draw.text((x + 34 * SS, legend_y + 21 * SS), "vortex ring", font=small, fill=DIM)

    # ---- readouts
    y = 606 * SS
    draw.text((x, y), state["phase"], font=mono, fill=state["phase_colour"])
    rows = [
        ("t*", "%.9f" % state["time"]),
        ("V / V0", "%.6e" % (state["volume"] / V0)),
        ("volume", "%.6e" % state["volume"]),
        ("centroid z", "%.6f" % state["centroid"]),
        ("max |u|", "%.4g" % state["maxu"]),
    ]
    if len(state["components"]) > 1:
        sizes = sorted((component[1] for component in state["components"]), reverse=True)
        rows.append(("larger / smaller V", "%.3e / %.3e" % (sizes[0], sizes[1])))
        rows.append(("removed so far", "none - both kept"))
    elif state["merged_info"]:
        info = state["merged_info"]
        rows.append(("merged at gap/h", "%.3f" % info.get("ratio", 0.0)))
        rows.append(("circulation after merge", "%+.2f of %+.2f"
                     % (info.get("merged", 0.0), info.get("sum", 0.0))))
    else:
        rows.append((state["removal_label"], "%.2f %% of V" % (100.0 * state["dropped_fraction"])))
        rows.append(("removed circulation", "%+.2f of %+.2f" % (state["dropped"], state["parent"])))
    for offset, (name, value) in enumerate(rows):
        row_y = y + (34 + offset * 26) * SS
        colour = WARN if offset >= 5 and (state["flagged"] or state["degenerate"]) else INK
        draw.text((x, row_y), name, font=small, fill=FAINT)
        draw.text((1520 * SS, row_y - 2 * SS), value, font=body, fill=colour, anchor="ra")

    draw.text((x, 824 * SS), "generation %d of %d   -   state %d / %d"
              % (state["generation"], state["generations"], index + 1, count),
              font=small, fill=FAINT)
    bar = (x, 852 * SS, 1520 * SS, 862 * SS)
    draw.rounded_rectangle(bar, radius=5 * SS, fill=(28, 52, 69))
    draw.rounded_rectangle((bar[0], bar[1], bar[0] + (bar[2] - bar[0]) * (index + 1) / count, bar[3]),
                           radius=5 * SS, fill=(21, 190, 198))

    if state["degenerate"] or state["overdriven"]:
        draw.rounded_rectangle((134 * SS, 130 * SS, 1006 * SS, 156 * SS), radius=6 * SS,
                               fill=(58, 22, 22), outline=WARN, width=SS)
        message = ("beyond the physical range - peak speed %.1e, clock advancing at round-off"
                   % state["maxu"]) if state["degenerate"] else (
                  "over-driven - the surface potential inherited from the surgeries is too "
                  "large; a rebound cannot pass V/V0 = 1")
        draw.text((146 * SS, 134 * SS), message, font=small, fill=WARN)

    if state["caption"]:
        draw.text((120 * SS, 852 * SS), state["caption"],
                  font=body, fill=WARN if (state["flagged"] or state["degenerate"]) else ACCENT)

    return image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


# ----------------------------------------------------------------------- main
def build_states():
    states = []
    for number, directory in enumerate(GENERATIONS, 1):
        frames, diag, log = load_generation(directory)
        keep = sample_evenly_in_time(diag, SAMPLE_TARGET.get(number, len(diag)))
        flagged = abs(log["dropped"]) > CIRCULATION_WARN * abs(log["parent"])
        for position, index in enumerate(keep):
            row = diag[index]
            previous_count = len(frames[keep[position - 1]]) // 2 if position else 1
            # The removal in force here: the one made at the split, or the most
            # recent mid-run reduction, whichever last happened.
            applied = dict(fraction=log["dropped_fraction"], gamma=log["dropped"],
                           label="discarded at split")
            overdriven = row[6] > VOLUME_TRUST
            absorbed = None
            if position:
                for key in sorted(log["absorb"]):
                    if int(diag[keep[position - 1]][1]) < key <= int(row[1]):
                        absorbed = log["absorb"][key]
            merged_info = None
            for key in sorted(log["merges"]):
                if key <= int(row[1]):
                    merged_info = log["merges"][key]
            for key in sorted(log["retirements"]):
                if key <= int(row[1]) and log["retirements"][key]["ratio"] > 0.0:
                    event = log["retirements"][key]
                    applied = dict(fraction=event["fraction"], gamma=event["gamma"],
                                   label="removed at reduction")
            components = [(frames[index][0:2], row[4], (row[12], row[13]))]
            if len(frames[index]) == 4:
                components.append((frames[index][2:4], row[5], (row[14], row[15])))
            caption = ""
            if position == 0:
                phase, colour = "split - smaller daughter discarded", ACCENT
                caption = ("generation %d: smaller daughter was %.2f %% of the volume - discarded, "
                           "carrying circulation %+.2f"
                           % (number, 100.0 * log["dropped_fraction"], log["dropped"]))
                if log["kept_both"]:
                    phase = "split - both daughters kept"
                    caption = ("generation %d: the daughters are comparable (%.1f %% and %.1f %% of V) "
                               "- both kept and evolved"
                               % (number, 100.0 * min(row[4], row[5]) / row[3],
                                  100.0 * max(row[4], row[5]) / row[3]))
                if log["degenerate"]:
                    caption = ("generation %d restarts inside the previous runaway - peak speed is already "
                               "%.1e and the clock barely advances; shown for the topology only"
                               % (number, row[16]))
                if flagged:
                    caption = ("generation %d discards circulation %+.2f against a parent %+.2f - past the "
                               "regime where dropping a daughter is a small correction"
                               % (number, log["dropped"], log["parent"]))
            elif previous_count > 1 and len(components) == 1:
                phase, colour = "coalesced", (255, 150, 120)
                event = None
                for key in sorted(log["merges"]):
                    if keep[position - 1] < key <= int(row[1]) or event is None:
                        event = log["merges"][key]
                caption = "the two components touched and were merged into one loop"
                if event:
                    caption = ("components touched at gap/h = %.3f and merged into one loop - "
                               "circulation %+.2f from a sum of %+.2f, volume change %+.3f %%"
                               % (event.get("ratio", 0.0), event.get("merged", 0.0),
                                  event.get("sum", 0.0), 100.0 * event.get("dv", 0.0)))
            elif absorbed:
                phase, colour = "pinch absorbed", (255, 150, 120)
                caption = ("a point of the old hole wall was pressed onto the axis - the closure "
                           "took it in, dropping %.3f %% of the volume"
                           % (100.0 * absorbed["fraction"]))
            elif log["axis"] and int(row[1]) >= log["axis"]["post"]:
                phase, colour = "whole bubble", BUBBLE_EDGE
                if position and int(row[1]) >= log["axis"]["post"] > int(diag[keep[position - 1]][1]):
                    ax = log["axis"]
                    phase, colour = "made whole again", (255, 150, 120)
                    caption = ("the hole reached %.3f R0 of the axis - torus made whole again, "
                               "releasing circulation %+.2f, volume change %+.2f %%"
                               % (ax.get("rmin", 0.0), ax.get("gamma", 0.0), 100.0 * ax.get("dv", 0.0)))
            elif len(components) > 1:
                phase, colour = "two daughters evolving", BUBBLE_EDGE
            else:
                phase, colour = "single daughter", BUBBLE_EDGE
            if index == len(frames) - 1:
                phase, colour = log["ending"], ACCENT
                last_generation = number == len(GENERATIONS)
                caption = {
                    "contact captured":
                        ("final captured contact - the next split produces a degenerate loop"
                         if last_generation
                         else "pre-contact checkpoint written - it splits into the next generation"),
                    "peak speed runs away":
                        "surface speed leaves the physical range and the step collapses - continuation ends here",
                    "time step underflow":
                        "the time step fell below the floor - the clock stops advancing usefully here",
                    "surface reached the axis":
                        "the growing bubble drove its lower surface onto the axis - the run ends here",
                    "expansion over-driven":
                        "shown to here only - past this the bubble over-expands and the result is not physical",
                }.get(log["ending"], "the continuation ends here")
            states.append(dict(
                components=components, time=row[2], volume=row[3], centroid=row[8],
                maxu=row[16], phase=phase, phase_colour=colour,
                caption=caption, generation=number, generations=len(GENERATIONS),
                dropped=applied["gamma"], parent=log["parent"],
                dropped_fraction=applied["fraction"], removal_label=applied["label"],
                merged_info=merged_info, flagged=flagged,
                degenerate=log["degenerate"], overdriven=overdriven,
                hold=16 if caption else 2,
            ))
    return states


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    states = build_states()
    print("rendering %d states" % len(states))

    writer = imageio_ffmpeg.write_frames(
        str(OUTPUT / "split_continuation.mp4"), (WIDTH, HEIGHT),
        fps=FPS, quality=8, macro_block_size=8,
    )
    writer.send(None)
    written, preview, image = 0, None, None
    for index, state in enumerate(states):
        image = render(state, index, len(states))
        raw = image.tobytes()
        repeats = state["hold"] if index else FPS
        if index == len(states) - 1:
            repeats = FPS * 2
        for _ in range(repeats):
            writer.send(raw)
            written += 1
        if state["caption"] and preview is None and index:
            preview = image.copy()
        if index % 100 == 0:
            print("  state %d/%d" % (index + 1, len(states)))
    writer.close()

    (preview or image).save(OUTPUT / "split_continuation_preview.png")
    print("wrote %d frames (%.1f s) to %s" % (written, written / FPS, OUTPUT / "split_continuation.mp4"))
    print("time span t* = %.9f .. %.9f" % (states[0]["time"], states[-1]["time"]))


if __name__ == "__main__":
    main()
