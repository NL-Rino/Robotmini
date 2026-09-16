# -*- coding: utf-8 -*-
"""Ghi file PNG khong can thu vien ngoai (chi zlib co san trong Python)."""

import struct
import zlib


def write_png(path, rows):
    """`rows` la danh sach cac hang, moi hang la bytes RGB dai 3*w."""
    h = len(rows)
    w = len(rows[0]) // 3
    raw = b"".join(b"\x00" + bytes(r) for r in rows)

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    out = [b"\x89PNG\r\n\x1a\n",
           chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)),
           chunk(b"IDAT", zlib.compress(raw, 6)),
           chunk(b"IEND", b"")]
    with open(path, "wb") as fh:
        fh.write(b"".join(out))
    return path


def save_image(path, img, scale=1):
    """`img` la tensor (3,h,w) trong [0,1]. `scale` de phong to cho de nhin."""
    a = (img.detach().clamp(0, 1) * 255).to("cpu").byte().permute(1, 2, 0)
    h, w, _ = a.shape
    if scale > 1:
        a = a.repeat_interleave(scale, 0).repeat_interleave(scale, 1)
        h, w = h * scale, w * scale
    flat = a.reshape(h, w * 3).numpy()
    return write_png(path, [bytes(flat[i].tolist()) for i in range(h)])


def save_grid(path, imgs, cols=4, scale=2, pad=2):
    """Xep nhieu anh thanh mot luoi."""
    import torch
    n = imgs.shape[0]
    rows = (n + cols - 1) // cols
    c, h, w = imgs.shape[1:]
    sheet = torch.ones(c, rows * (h + pad) + pad, cols * (w + pad) + pad)
    for i in range(n):
        r, cc = divmod(i, cols)
        y0 = pad + r * (h + pad)
        x0 = pad + cc * (w + pad)
        sheet[:, y0:y0 + h, x0:x0 + w] = imgs[i].detach().cpu()
    return save_image(path, sheet, scale=scale)
