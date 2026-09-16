# -*- coding: utf-8 -*-
"""Do thong luong that tren may dang chay, roi quy ra tien.

    python -m v2.tools.bench                  # hoi chon GPU
    python -m v2.tools.bench --device cuda:0 --yes
    python -m v2.tools.bench --gia 18500      # dong moi gio

Chay cai nay TRUOC khi thue dai han. No cho biet may do ve duoc bao nhieu
khung hinh moi giay, va bao nhieu tien cho mot trieu buoc.
"""

import argparse
import sys
import time

import torch

from .. import device as dev_mod
from .. import params as P
from ..env import FleetEnv3D
from ..policy import ActorCritic
from ..ppo import PPO, Buffer, collect

# Gia thue thang 9/2026, dong moi gio
GIA = {"NVIDIA-RTX-A4000": 14300, "NVIDIA-RTX-3090": 18500}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timeit(fn, device, n=10, warm=3):
    for _ in range(warm):
        fn()
    sync(device)
    t0 = time.time()
    for _ in range(n):
        fn()
    sync(device)
    return (time.time() - t0) / n


def bench(device, envs, T=64, verbose=True):
    env = FleetEnv3D(n_envs=envs, device=device, seed=0, max_steps=600)
    model = ActorCritic().to(device)
    algo = PPO(model, device, minibatches=4, epochs=4,
               amp=device.type == "cuda")
    buf = Buffer(T, envs, (3, P.CAM_H, P.CAM_W), P.N_SCALARS, 3, device)
    state = dict(h=model.initial_state(envs, device),
                 done=torch.zeros(envs, device=device))
    a = torch.zeros(envs, 3, device=device)

    t_obs = timeit(lambda: env.observe(), device, n=8)
    t_step = timeit(lambda: env.step(a), device, n=8)
    with torch.no_grad():
        img, sca = env.observe()
        h = model.initial_state(envs, device)
        t_pol = timeit(lambda: model.act(img, sca, h), device, n=8)

    sync(device)
    t0 = time.time()
    last_v = collect(env, model, buf, state, device,
                     amp=device.type == "cuda")
    sync(device)
    t_col = time.time() - t0
    t0 = time.time()
    algo.update(buf, last_v)
    sync(device)
    t_upd = time.time() - t0

    sps = envs * T / (t_col + t_upd)
    mem = (torch.cuda.max_memory_allocated(device) / 1024 ** 3
           if device.type == "cuda" else 0.0)
    if verbose:
        print(f"  {envs:4d} the gioi | ve+cam bien {t_obs * 1e3:7.1f} ms | "
              f"vat ly {t_step * 1e3:6.1f} ms | bo nao {t_pol * 1e3:6.1f} ms | "
              f"thu thap {t_col:5.2f}s hoc {t_upd:5.2f}s | "
              f"{sps:8.0f} buoc/s | VRAM {mem:5.2f} GB", flush=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    return sps, mem, t_obs, t_pol


def main(argv=None):
    ap = argparse.ArgumentParser(description="Do thong luong va quy ra tien")
    dev_mod.add_argument(ap)
    ap.add_argument("--envs", type=int, nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--gia", type=int, default=None, help="dong moi gio")
    ap.add_argument("--muc-tieu", type=float, default=50.0,
                    help="so trieu buoc muon chay")
    a = ap.parse_args(argv)
    device = dev_mod.from_args(a)

    name = (torch.cuda.get_device_name(device) if device.type == "cuda"
            else "CPU")
    print(f"\nmay: {name}")
    if device.type == "cuda":
        free, tot = torch.cuda.mem_get_info(device)
        print(f"VRAM: {tot / 1024 ** 3:.0f} GB, con trong {free / 1024 ** 3:.0f} GB")
    print()

    sizes = a.envs or ([16, 32, 64] if device.type == "cpu"
                       else [64, 128, 256, 512, 1024])
    best = (0.0, 0)
    for n in sizes:
        try:
            sps, mem, _o, _p = bench(device, n, a.steps)
        except RuntimeError as e:
            print(f"  {n:4d} the gioi | KHONG DU BO NHO ({str(e)[:40]}...)")
            break
        if sps > best[0]:
            best = (sps, n)
    sps, n = best
    if sps <= 0:
        return 1

    print(f"\nnhanh nhat: {n} the gioi song song, {sps:,.0f} buoc/giay"
          .replace(",", "."))
    gia = a.gia
    if gia is None:
        for k, v in GIA.items():
            if k.split("-")[-1].lower() in name.lower().replace(" ", ""):
                gia = v
    target = a.muc_tieu * 1e6
    hours = target / sps / 3600.0
    print(f"chay {a.muc_tieu:.0f} trieu buoc het {hours:.1f} gio")
    if gia:
        print(f"gia {gia:,} d/gio -> {hours * gia:,.0f} d cho ca lan chay"
              .replace(",", "."))
        print(f"tuc la {gia / sps * 1e6 / 3600:,.0f} d moi trieu buoc"
              .replace(",", "."))
    else:
        print("(khong biet gia may nay, dung --gia de tinh tien)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
