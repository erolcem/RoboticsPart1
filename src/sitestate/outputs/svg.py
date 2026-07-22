"""Small SVG helpers for grid maps and trajectories in the HTML report."""

from __future__ import annotations

import numpy as np


def grid_svg(
    values: np.ndarray,
    colors: dict[int, str],
    x0: float,
    y0: float,
    res: float,
    px_per_m: float = 30.0,
    overlays: str = "",
) -> str:
    """Render an int-coded grid as run-length row rects (small file size).

    Grid rows are y-up in world coordinates; SVG is y-down, so flip.
    """
    ny, nx = values.shape
    w_px, h_px = nx * res * px_per_m, ny * res * px_per_m
    cell_px = res * px_per_m
    rects: list[str] = []
    for j in range(ny):
        row = values[j]
        i = 0
        while i < nx:
            v = int(row[i])
            k = i
            while k < nx and int(row[k]) == v:
                k += 1
            color = colors.get(v)
            if color:
                x_px = i * cell_px
                y_px = (ny - 1 - j) * cell_px
                rects.append(
                    f'<rect x="{x_px:.1f}" y="{y_px:.1f}" '
                    f'width="{(k - i) * cell_px:.1f}" height="{cell_px + 0.3:.1f}" '
                    f'fill="{color}"/>'
                )
            i = k
    return (
        f'<svg viewBox="0 0 {w_px:.0f} {h_px:.0f}" width="{w_px:.0f}" '
        f'xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;'
        f'background:#fafafa;border:1px solid #ddd;border-radius:6px">'
        f"{''.join(rects)}{overlays}</svg>"
    )


def world_to_px(
    x: float, y: float, x0: float, y0: float, ny: int, res: float, px_per_m: float = 30.0
) -> tuple[float, float]:
    return (x - x0) * px_per_m, (ny * res - (y - y0)) * px_per_m


def trajectory_overlay(
    poses: np.ndarray, x0: float, y0: float, ny: int, res: float, px_per_m: float = 30.0,
    color: str = "#2563eb",
) -> str:
    pts = " ".join(
        f"{p[0]:.1f},{p[1]:.1f}"
        for p in (world_to_px(x, y, x0, y0, ny, res, px_per_m) for x, y, *_ in poses)
    )
    start = world_to_px(poses[0][0], poses[0][1], x0, y0, ny, res, px_per_m)
    return (
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-opacity="0.85"/>'
        f'<circle cx="{start[0]:.1f}" cy="{start[1]:.1f}" r="4" fill="{color}"/>'
    )


def bbox_overlay(
    bbox: list[float], x0: float, y0: float, ny: int, res: float, px_per_m: float = 30.0,
    color: str = "#dc2626", label: str = "",
) -> str:
    xa, ya = world_to_px(bbox[0], bbox[3], x0, y0, ny, res, px_per_m)  # top-left in svg
    xb, yb = world_to_px(bbox[2], bbox[1], x0, y0, ny, res, px_per_m)
    text = (
        f'<text x="{xa:.1f}" y="{ya - 4:.1f}" font-size="11" fill="{color}" '
        f'font-family="sans-serif">{label}</text>'
        if label
        else ""
    )
    return (
        f'<rect x="{xa:.1f}" y="{ya:.1f}" width="{xb - xa:.1f}" height="{yb - ya:.1f}" '
        f'fill="none" stroke="{color}" stroke-width="2.5" rx="3"/>{text}'
    )


def markers_overlay(
    points: dict[str, tuple[float, float]], x0: float, y0: float, ny: int, res: float,
    px_per_m: float = 30.0, color: str = "#7c3aed",
) -> str:
    parts = []
    for name, (x, y) in points.items():
        px, py = world_to_px(x, y, x0, y0, ny, res, px_per_m)
        parts.append(
            f'<g><circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="none" stroke="{color}" '
            f'stroke-width="2"/><line x1="{px - 7:.1f}" y1="{py:.1f}" x2="{px + 7:.1f}" '
            f'y2="{py:.1f}" stroke="{color}" stroke-width="1"/>'
            f'<line x1="{px:.1f}" y1="{py - 7:.1f}" x2="{px:.1f}" y2="{py + 7:.1f}" '
            f'stroke="{color}" stroke-width="1"/>'
            f'<text x="{px + 8:.1f}" y="{py - 6:.1f}" font-size="11" fill="{color}" '
            f'font-family="sans-serif">{name}</text></g>'
        )
    return "".join(parts)
