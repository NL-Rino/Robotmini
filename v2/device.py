# -*- coding: utf-8 -*-
"""Chon may tinh de chay: GPU nao, hay CPU.

Chay tren may thue co nhieu card thi hoi chon. Chay o nha khong co card thi
van chay duoc bang CPU - cham hon nhieu nhung van xem duoc no lam gi.
"""

import os
import sys

import torch


def list_devices():
    """Danh sach may tinh dung duoc, kem bo nho."""
    out = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            out.append(dict(key=f"cuda:{i}", name=p.name,
                            vram_gb=p.total_memory / (1024 ** 3),
                            sm=p.multi_processor_count))
    out.append(dict(key="cpu", name="CPU (" + (os.cpu_count() and
                    f"{os.cpu_count()} luong" or "?") + ")",
                    vram_gb=0.0, sm=0))
    return out


def describe(d):
    if d["key"] == "cpu":
        return f"{d['name']} - chay duoc nhung cham, chi de xem thu"
    return (f"{d['name']}  {d['vram_gb']:.0f} GB VRAM, "
            f"{d['sm']} cum tinh toan")


def pick(prefer=None, interactive=None, quiet=False):
    """Tra ve torch.device.

    prefer      : "cuda", "cuda:1", "cpu", hoac None
    interactive : None = tu quyet (chi hoi khi co man hinh va co nhieu lua chon)
    """
    devs = list_devices()
    by_key = {d["key"]: d for d in devs}

    if prefer:
        p = str(prefer).lower()
        if p == "cuda":
            p = "cuda:0"
        if p in by_key:
            if not quiet:
                print(f"chay tren {describe(by_key[p])}")
            return torch.device(p)
        if not quiet:
            print(f"khong co '{prefer}', chuyen sang lua chon khac")

    gpus = [d for d in devs if d["key"] != "cpu"]
    if not gpus:
        if not quiet:
            print("KHONG THAY GPU NAO. Chay bang CPU - cham hon vai chuc lan,")
            print("du de xem xe lam gi nhung dung de huan luyen that.")
        return torch.device("cpu")

    if interactive is None:
        interactive = sys.stdin is not None and sys.stdin.isatty()
    if not interactive or len(devs) == 2:
        # Chi co mot card: khoi hoi.
        if not quiet:
            print(f"chay tren {describe(gpus[0])}")
        return torch.device(gpus[0]["key"])

    print("\nChon may tinh de chay:")
    for i, d in enumerate(devs):
        print(f"  [{i}] {describe(d)}")
    while True:
        try:
            raw = input(f"so thu tu [0-{len(devs) - 1}], Enter = 0: ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = ""
        if raw == "":
            raw = "0"
        if raw.isdigit() and 0 <= int(raw) < len(devs):
            d = devs[int(raw)]
            print(f"chay tren {describe(d)}\n")
            return torch.device(d["key"])
        print("  khong hop le, nhap lai")


def add_argument(parser):
    parser.add_argument("--device", default=None,
                        help="cuda / cuda:1 / cpu. Bo trong thi hoi hoac tu chon")
    parser.add_argument("--yes", action="store_true",
                        help="khong hoi gi ca, tu chon card dau tien")


def from_args(a, quiet=False):
    return pick(prefer=getattr(a, "device", None),
                interactive=(False if getattr(a, "yes", False) else None),
                quiet=quiet)
