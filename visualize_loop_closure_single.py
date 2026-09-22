#!/usr/bin/env python3
"""
单轨迹回环可视化（顶刊/顶会版式）：
- 仅绘制一条轨迹；正确匹配时，对匹配对两端（query / target 帧）各高亮一段轨迹：
  以该帧为中心，前后各延伸 highlight_length 个索引（闭区间，与 merge 后可能相连）。
- 默认无坐标轴，便于论文拼图；可选矢量/高分辨率位图导出
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Publication-style defaults (CVPR/ICCV/TPAMI / Nature-style figures)
# ---------------------------------------------------------------------------
def _setup_publication_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "lines.antialiased": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


# Wong-style colorblind-safe palette
COLOR_TRAJECTORY = "#4C72B0"  # muted blue
COLOR_HIGHLIGHT = "#009E73"   # bluish green (distinct from blue)


def parse_result_file(filename: str) -> Tuple[Dict[int, np.ndarray], List[Tuple[int, int, float, Optional[int]]]]:
    """
    解析结果文件：
    - Trajectory 段：frame_id x y z ...
    - Loop Closures 段：frame_id1 frame_id2 [match_score] ... [is_correct(0/1)]
    """
    trajectory: Dict[int, np.ndarray] = {}
    loop_closures: List[Tuple[int, int, float, Optional[int]]] = []

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            if "Trajectory" in line:
                section = "trajectory"
            elif "Loop Closures" in line:
                section = "loop_closures"
            continue

        parts = line.split()
        if section == "trajectory":
            if len(parts) < 4:
                continue
            try:
                frame_id = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                trajectory[frame_id] = np.array([x, y, z], dtype=float)
            except ValueError:
                continue
        elif section == "loop_closures":
            if len(parts) < 2:
                continue
            try:
                frame_id1 = int(parts[0])
                frame_id2 = int(parts[1])
                match_score = float(parts[2]) if len(parts) > 2 else 0.0
                is_correct = None
                if parts[-1] in ("0", "1"):
                    is_correct = int(parts[-1])
                loop_closures.append((frame_id1, frame_id2, match_score, is_correct))
            except ValueError:
                continue

    return trajectory, loop_closures


def _index_window(idx: int, radius: int, n_frames: int) -> Tuple[int, int]:
    """Closed index interval [start, end] centered near idx, ±radius, clipped to trajectory."""
    start = max(0, idx - radius)
    end = min(n_frames - 1, idx + radius)
    return (start, end)


def collect_green_ranges(
    frame_ids: List[int],
    trajectory: Dict[int, np.ndarray],
    loop_closures: List[Tuple[int, int, float, Optional[int]]],
    highlight_length: int,
    fallback_distance_threshold: float,
) -> List[Tuple[int, int]]:
    """
    For each correct loop closure (id1, id2), add two highlight intervals on the trajectory:
    one around the query frame id1 and one around the target frame id2 (symmetric ±highlight_length).
    """
    frame_to_idx = {fid: i for i, fid in enumerate(frame_ids)}
    ranges: List[Tuple[int, int]] = []
    correct_count = 0
    n = len(frame_ids)

    for frame_id1, frame_id2, _score, is_correct_flag in loop_closures:
        if frame_id1 not in trajectory or frame_id2 not in trajectory:
            continue
        if frame_id1 not in frame_to_idx or frame_id2 not in frame_to_idx:
            continue

        if is_correct_flag is None:
            dist = np.linalg.norm(trajectory[frame_id1] - trajectory[frame_id2])
            is_correct = dist <= fallback_distance_threshold
        else:
            is_correct = (is_correct_flag == 1)

        if not is_correct:
            continue

        correct_count += 1
        idx1 = frame_to_idx[frame_id1]
        idx2 = frame_to_idx[frame_id2]
        ranges.append(_index_window(idx1, highlight_length, n))
        ranges.append(_index_window(idx2, highlight_length, n))

    print(f"Correct matches used for highlighting: {correct_count}")
    return ranges


def merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    ranges = sorted(ranges, key=lambda x: x[0])
    merged = [ranges[0]]
    for s, e in ranges[1:]:
        ps, pe = merged[-1]
        if s <= pe + 1:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def visualize_single_trajectory(
    trajectory: Dict[int, np.ndarray],
    loop_closures: List[Tuple[int, int, float, Optional[int]]],
    output_file: Optional[str] = None,
    highlight_length: int = 10,
    fallback_distance_threshold: float = 20.0,
    base_color: str = COLOR_TRAJECTORY,
    highlight_color: str = COLOR_HIGHLIGHT,
    base_linewidth: float = 1.8,
    highlight_linewidth: float = 3.6,
    init_elev: Optional[float] = None,
    init_azim: Optional[float] = None,
    init_dist: Optional[float] = None,
    figsize: Tuple[float, float] = (4.25, 3.6),
    title: Optional[str] = None,
    show_title: bool = False,
    legend_loc: str = "upper right",
    save_pdf: bool = True,
):
    """
    Single 3D trajectory with highlighted segments for correct loop detections.

    Publication defaults: sans-serif, 600 DPI raster, optional companion PDF.
    """
    if not trajectory:
        print("Error: No trajectory data found!")
        return

    _setup_publication_style()

    frame_ids = sorted(trajectory.keys())
    pts = np.array([trajectory[fid] for fid in frame_ids], dtype=float)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    if init_elev is not None or init_azim is not None:
        elev = init_elev if init_elev is not None else ax.elev
        azim = init_azim if init_azim is not None else ax.azim
        ax.view_init(elev=elev, azim=azim)
    if init_dist is not None and hasattr(ax, "dist"):
        ax.dist = init_dist

    ax.plot(
        x,
        y,
        z,
        color=base_color,
        linewidth=base_linewidth,
        alpha=0.85,
        label="Trajectory",
        solid_capstyle="round",
        solid_joinstyle="round",
    )

    ranges = collect_green_ranges(
        frame_ids, trajectory, loop_closures, highlight_length, fallback_distance_threshold
    )
    merged = merge_ranges(ranges)
    for s, e in merged:
        if e <= s:
            continue
        ax.plot(
            x[s : e + 1],
            y[s : e + 1],
            z[s : e + 1],
            color=highlight_color,
            linewidth=highlight_linewidth,
            alpha=0.98,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    print(f"Highlighted trajectory segments: {len(merged)}")

    legend_elements = [
        Line2D(
            [0],
            [0],
            color=base_color,
            linewidth=base_linewidth,
            alpha=0.85,
            label="Trajectory",
            solid_capstyle="round",
        ),
        Line2D(
            [0],
            [0],
            color=highlight_color,
            linewidth=highlight_linewidth,
            alpha=0.98,
            label="Correct loop",
            solid_capstyle="round",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc=legend_loc,
        frameon=True,
        framealpha=1.0,
        fancybox=False,
        edgecolor="#d0d0d0",
        borderpad=0.45,
        handlelength=2.2,
        handletextpad=0.55,
    )

    ax.set_axis_off()
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("none")

    if show_title and title:
        ax.set_title(title, pad=6)
    elif show_title and not title:
        ax.set_title("Loop closure highlight", pad=6)

    plt.tight_layout(pad=0.2)

    try:
        print("Current 3D view:")
        print(f"  elev: {getattr(ax, 'elev', None)}")
        print(f"  azim: {getattr(ax, 'azim', None)}")
        print(f"  dist: {getattr(ax, 'dist', None)}")
    except Exception as exc:
        print(f"Warning: failed to query current view parameters: {exc}")

    def _save(path: str, dpi: int):
        plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")

    if output_file:
        root, ext = os.path.splitext(output_file)
        ext = ext.lower()
        if ext in (".pdf", ".svg", ".eps"):
            _save(output_file, dpi=600)
            print(f"Saved (vector): {output_file}")
        else:
            if not ext:
                output_file = output_file + ".png"
                ext = ".png"
            _save(output_file, dpi=600)
            print(f"Saved (raster): {output_file}")
            if save_pdf and ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                pdf_path = root + ".pdf"
                _save(pdf_path, dpi=600)
                print(f"Saved (companion PDF): {pdf_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Single-trajectory loop-closure visualization (publication style)."
    )
    parser.add_argument("input_file", type=str, help="Input result file from loop closure detection")
    parser.add_argument("--output", type=str, default=None, help="Output path (.png / .pdf / .svg)")
    parser.add_argument(
        "--highlight-length",
        type=int,
        default=10,
        help="Radius in frames: around each of query/target frames, highlight ±N indices (default: 10)",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=20.0,
        help="Pose-distance threshold if is_correct is missing (default: 20.0)",
    )
    parser.add_argument("--elev", type=float, default=None, help="Initial elevation (deg)")
    parser.add_argument("--azim", type=float, default=None, help="Initial azimuth (deg)")
    parser.add_argument("--dist", type=float, default=None, help="Initial camera distance (mpl 3D)")
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Figure title (only if --show-title)",
    )
    parser.add_argument(
        "--show-title",
        action="store_true",
        help="Show title (default: off, for paper figures)",
    )
    parser.add_argument(
        "--no-companion-pdf",
        action="store_true",
        help="When saving PNG, do not also write a same-name .pdf",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[4.25, 3.6],
        metavar=("W", "H"),
        help="Figure size in inches (default: 4.25 3.6, single-column friendly)",
    )
    args = parser.parse_args()

    print(f"Reading result file: {args.input_file}")
    trajectory, loop_closures = parse_result_file(args.input_file)
    print(f"Loaded {len(trajectory)} trajectory points")
    print(f"Found {len(loop_closures)} loop closures")

    if not trajectory:
        print("Error: No trajectory data found in the file!")
        sys.exit(1)

    visualize_single_trajectory(
        trajectory=trajectory,
        loop_closures=loop_closures,
        output_file=args.output,
        highlight_length=args.highlight_length,
        fallback_distance_threshold=args.distance_threshold,
        init_elev=args.elev,
        init_azim=args.azim,
        init_dist=args.dist,
        figsize=(args.figsize[0], args.figsize[1]),
        title=args.title,
        show_title=args.show_title,
        save_pdf=not args.no_companion_pdf,
    )
    print("Done.")


if __name__ == "__main__":
    main()
