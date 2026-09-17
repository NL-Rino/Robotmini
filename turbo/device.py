# -*- coding: utf-8 -*-
"""Chon may tinh de chay: GPU nao, hay CPU.

Co mot ban giong the trong `v2/`. Chep chu khong dung chung la co y: du an
v2 (con xe co camera) phai thao ra duoc ma khong lam hong ban v1 nay, va
nguoc lai. Chin muoi dong lap lai re hon mot rang buoc giua hai du an.
"""

import os
import sys

import torch


def _directml():
    """Card chay qua DirectML - duong duy nhat cho card LIEN tren Windows.

    Intel HD 620, UHD 620, Iris, va ca card AMD tich hop deu khong co CUDA.
    Chung co DirectX 12, va `torch-directml` bien DirectX 12 thanh mot may
    tinh cua PyTorch. Cai rieng: `pip install torch-directml` (no keo theo
    ban torch rieng cua no, nen nen dung mot moi truong ao rieng).
    """
    try:
        import torch_directml as dml
    except Exception:
        return []
    try:
        if not dml.is_available():
            return []
        return [dict(key=f"dml:{i}", name=dml.device_name(i),
                     vram_gb=0.0, sm=0, kind="dml", index=i)
                for i in range(dml.device_count())]
    except Exception:
        return []


def _xpu():
    """Card Intel doi moi (Arc, Core Ultra) qua duong XPU cua PyTorch.

    KHONG chay cho HD 620: doi may do la Gen9, con XPU can Gen12 tro len.
    """
    try:
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            return []
        out = []
        for i in range(torch.xpu.device_count()):
            try:
                p = torch.xpu.get_device_properties(i)
                vram = getattr(p, "total_memory", 0) / (1024 ** 3)
                name = getattr(p, "name", f"Intel XPU {i}")
            except Exception:
                vram, name = 0.0, f"Intel XPU {i}"
            out.append(dict(key=f"xpu:{i}", name=name, vram_gb=vram, sm=0,
                            kind="xpu", index=i))
        return out
    except Exception:
        return []


def _mps():
    try:
        if torch.backends.mps.is_available():
            return [dict(key="mps", name="GPU cua Mac (Metal)", vram_gb=0.0,
                         sm=0, kind="mps", index=0)]
    except Exception:
        pass
    return []


def list_devices():
    out = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            out.append(dict(key=f"cuda:{i}", name=p.name, kind="cuda", index=i,
                            vram_gb=p.total_memory / (1024 ** 3),
                            sm=p.multi_processor_count))
    out += _xpu() + _directml() + _mps()
    out.append(dict(key="cpu", vram_gb=0.0, sm=0, kind="cpu", index=0,
                    name=f"CPU ({os.cpu_count() or '?'} luong)"))
    return out


def resolve(key):
    """Doi khoa thanh `torch.device`. DirectML khong co ten trong torch."""
    k = str(key).lower()
    if k.startswith("dml"):
        import torch_directml as dml
        i = int(k.split(":")[1]) if ":" in k else 0
        return dml.device(i)
    if k == "cuda":
        k = "cuda:0"
    return torch.device(k)


def describe(d):
    if d["key"] == "cpu":
        return f"{d['name']} - chay duoc nhung cham, chi de xem thu"
    if d.get("kind") == "dml":
        return f"{d['name']} (DirectML) - card lien, dung chung RAM voi CPU"
    if d["vram_gb"] <= 0.0:
        return f"{d['name']}"
    return f"{d['name']}  {d['vram_gb']:.0f} GB VRAM, {d['sm']} cum tinh toan"


def pick(prefer=None, interactive=None, quiet=False):
    devs = list_devices()
    by_key = {d["key"]: d for d in devs}
    if prefer:
        p = str(prefer).lower()
        if p == "cuda":
            p = "cuda:0"
        if p == "dml":
            p = "dml:0"
        if p in by_key:
            if not quiet:
                print(f"chay tren {describe(by_key[p])}")
            return resolve(p)
        if not quiet:
            print(f"khong co '{prefer}', chuyen sang lua chon khac")

    gpus = [d for d in devs if d["key"] != "cpu"]
    if not gpus:
        if not quiet:
            print("KHONG THAY GPU NAO. Chay bang CPU - cham hon nhieu lan,")
            print("du de xem xe lam gi nhung dung de huan luyen that.")
        return torch.device("cpu")
    if interactive is None:
        interactive = sys.stdin is not None and sys.stdin.isatty()
    if not interactive or len(devs) == 2:
        if not quiet:
            print(f"chay tren {describe(gpus[0])}")
        return resolve(gpus[0]["key"])

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
        if raw.isdigit() and int(raw) < len(devs):
            d = devs[int(raw)]
            print(f"chay tren {describe(d)}")
            return resolve(d["key"])
        print("khong hop le, thu lai")


def engines():
    """Cac bo may chay co the chon, kem mot cau giai thich ngan.

    (nhan, module, --device, ghi chu). Ban `train.train` cham diem TUNG ca
    the o mot tien trinh rieng; ban `turbo.train` gop ca quan the vao mot
    phep tinh. Do tren may 4 loi: quan the 32 thi ban cu nhanh hon 1,1 lan,
    quan the 128 thi ban theo lo nhanh hon 1,2 lan - va chi ban theo lo moi
    dung duoc card do hoa.

    De o day chu khong o `app.py` de con kiem thu duoc: may nao khong co
    Tkinter (may chu thue chang han) van chay ham nay duoc.
    """
    tips = {"cuda": "card roi - de quan the that to (128, 256, 512)",
            "xpu": "card Intel doi moi - thu quan the 64 tro len",
            "dml": "card lien, dung chung RAM voi CPU - nhanh len it thoi",
            "cpu": "chi hon khi quan the tu 64 tro len"}
    out = [("tung xe mot - CPU, nhieu tien trinh", "train.train", None,
            "chac an; quan the nho thi day la nhanh nhat")]
    for d in list_devices():
        kind = d.get("kind", "cpu")
        label = "ca lo mot luc - " + ("CPU" if kind == "cpu" else d["name"])
        out.append((label, "turbo.train", d["key"], tips.get(kind, "")))
    # Card manh len dau - do la ly do ban theo lo ton tai. May khong co card
    # thi "tung xe mot" len dau, vi no van la cai nhanh hon o quan the nho.
    rank = {"cuda": 0, "xpu": 1, "mps": 1, "dml": 2, "cpu": 4}
    keys = {d["key"]: rank.get(d.get("kind", "cpu"), 4) for d in list_devices()}
    out.sort(key=lambda e: keys.get(e[2], 9) if e[2] else 3)
    return out


def add_argument(ap):
    ap.add_argument("--device", default=None,
                    help="cuda, cuda:1, cpu. Bo trong thi tu hoi/tu chon")
    ap.add_argument("--list-devices", action="store_true",
                    help="liet ke may tinh dung duoc roi thoat")


def from_args(a, quiet=False):
    if getattr(a, "list_devices", False):
        for d in list_devices():
            print(" ", describe(d))
        raise SystemExit(0)
    return pick(getattr(a, "device", None), quiet=quiet)
