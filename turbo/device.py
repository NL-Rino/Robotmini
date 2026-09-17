# -*- coding: utf-8 -*-
"""Chon may tinh de chay: GPU nao, hay CPU.

Co mot ban giong the trong `v2/`. Chep chu khong dung chung la co y: du an
v2 (con xe co camera) phai thao ra duoc ma khong lam hong ban v1 nay, va
nguoc lai. Chin muoi dong lap lai re hon mot rang buoc giua hai du an.
"""

import os
import sys

import torch


def list_devices():
    out = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            out.append(dict(key=f"cuda:{i}", name=p.name,
                            vram_gb=p.total_memory / (1024 ** 3),
                            sm=p.multi_processor_count))
    out.append(dict(key="cpu", vram_gb=0.0, sm=0,
                    name=f"CPU ({os.cpu_count() or '?'} luong)"))
    return out


def describe(d):
    if d["key"] == "cpu":
        return f"{d['name']} - chay duoc nhung cham, chi de xem thu"
    return f"{d['name']}  {d['vram_gb']:.0f} GB VRAM, {d['sm']} cum tinh toan"


def pick(prefer=None, interactive=None, quiet=False):
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
            print("KHONG THAY GPU NAO. Chay bang CPU - cham hon nhieu lan,")
            print("du de xem xe lam gi nhung dung de huan luyen that.")
        return torch.device("cpu")
    if interactive is None:
        interactive = sys.stdin is not None and sys.stdin.isatty()
    if not interactive or len(devs) == 2:
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
        if raw.isdigit() and int(raw) < len(devs):
            d = devs[int(raw)]
            print(f"chay tren {describe(d)}")
            return torch.device(d["key"])
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
    out = [("tung xe mot - CPU, nhieu tien trinh", "train.train", None,
            "chac an; quan the nho thi day la nhanh nhat")]
    for d in list_devices():
        if d["key"] == "cpu":
            out.append(("ca lo mot luc - CPU", "turbo.train", "cpu",
                        "chi hon khi quan the tu 64 tro len"))
        else:
            out.append((f"ca lo mot luc - {d['name']}", "turbo.train", d["key"],
                        f"{d['vram_gb']:.0f} GB VRAM - de quan the that to"))
    # Co card thi de card len dau: do la ly do ban theo lo ton tai.
    out.sort(key=lambda e: 0 if (e[2] or "").startswith("cuda") else 1)
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
