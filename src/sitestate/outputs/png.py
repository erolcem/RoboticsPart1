"""Minimal pure-stdlib PNG encoder (zlib + struct, no external imaging
dependency) used to embed depth-frame evidence as data-URI images in the
HTML report and web viewer.
"""

from __future__ import annotations

import base64
import struct
import zlib

import numpy as np


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def encode_png(image: np.ndarray) -> bytes:
    """Encode an (H, W) grayscale or (H, W, 3) RGB uint8 array as PNG bytes."""
    img = np.asarray(image)
    if img.dtype != np.uint8:
        raise ValueError("encode_png expects uint8 data")
    if img.ndim == 2:
        color_type, channels = 0, 1
    elif img.ndim == 3 and img.shape[2] == 3:
        color_type, channels = 2, 3
    else:
        raise ValueError(f"unsupported image shape {img.shape}")
    h, w = img.shape[:2]
    raw = img.reshape(h, w * channels)
    # filter byte 0 (None) before each scanline
    scanlines = b"".join(b"\x00" + raw[j].tobytes() for j in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(scanlines, 6))
        + _chunk(b"IEND", b"")
    )


def data_uri(image: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png(image)).decode()


def depth_strip(depths: np.ndarray, max_range: float, height: int = 28) -> np.ndarray:
    """Render a 1D depth row as a viewable grayscale strip: near = bright."""
    d = np.clip(np.asarray(depths, dtype=float), 0.0, max_range)
    gray = (255 * (1.0 - d / max_range)).astype(np.uint8)
    return np.repeat(gray[None, :], height, axis=0)
