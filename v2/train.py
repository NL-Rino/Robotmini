# -*- coding: utf-8 -*-
"""Huan luyen bang PPO tren GPU.

    python -m v2.train                       # hoi chon GPU roi chay
    python -m v2.train --device cuda:0 --yes # khong hoi gi ca
    python -m v2.train --envs 512 --amp

Dung an toan: tao file STOP trong thu muc --out (hoac bam Ctrl-C mot lan).
Dang chay vong nao thi chay not vong do, luu checkpoint, roi thoat.
"""

import argparse
import json
import os
import shutil
import signal
import sys
import time

import torch

from . import device as dev_mod
from . import params as P
from .env import FleetEnv3D
from .policy import ActorCritic
from .ppo import PPO, Buffer, collect


def vram_budget(device):
    if device.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info(device)
    return free / (1024 ** 3), total / (1024 ** 3)


def suggest_envs(device, T):
    """Chon so moi truong cho vua VRAM.

    Bo dem anh la thu an VRAM nhieu nhat: T * B * 3*48*64 byte (uint8).
    Chua ke mang, gradient va bo dem trung gian, nen chi lay khoang mot phan
    ba cho so tro nho roi tru hao.
    """
    b = vram_budget(device)
    if b is None:
        return 32
    free_gb = b[0]
    per_env = T * 3 * P.CAM_H * P.CAM_W / (1024 ** 3)      # GB moi moi truong
    n = int((free_gb * 0.30) / max(per_env, 1e-9))
    return max(32, min(1024, (n // 32) * 32))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Nuoi bo nao co mat, tren GPU")
    dev_mod.add_argument(ap)
    ap.add_argument("--out", default="v2/runs/thu1")
    ap.add_argument("--envs", type=int, default=0, help="0 = tu chon theo VRAM")
    ap.add_argument("--steps", type=int, default=64, help="so buoc moi vong")
    ap.add_argument("--total-steps", type=int, default=50_000_000)
    ap.add_argument("--max-episode", type=int, default=600)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--ent", type=float, default=0.004)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--curriculum-steps", type=int, default=15_000_000)
    ap.add_argument("--amp", action="store_true", help="dung nua do chinh xac")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1)
    a = ap.parse_args(argv)

    device = dev_mod.from_args(a)
    os.makedirs(a.out, exist_ok=True)
    stop_path = os.path.join(a.out, "STOP")
    if os.path.exists(stop_path):
        os.remove(stop_path)
    status_path = os.path.join(a.out, "status.jsonl")

    n_envs = a.envs or suggest_envs(device, a.steps)
    use_amp = (a.amp or device.type == "cuda") and not a.no_amp
    torch.manual_seed(a.seed)

    env = FleetEnv3D(n_envs=n_envs, device=device, seed=a.seed,
                     max_steps=a.max_episode)
    model = ActorCritic().to(device)
    algo = PPO(model, device, lr=a.lr, clip=a.clip, epochs=a.epochs,
               minibatches=a.minibatches, ent_coef=a.ent, amp=use_amp)
    buf = Buffer(a.steps, n_envs, (3, P.CAM_H, P.CAM_W), P.N_SCALARS, 3, device)
    state = dict(h=model.initial_state(n_envs, device),
                 done=torch.zeros(n_envs, device=device))

    done_steps = 0
    it0 = 0
    if a.resume:
        path = a.resume
        if os.path.isdir(path):
            path = os.path.join(path, "state.pt")
        if os.path.exists(path):
            ck = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            algo.opt.load_state_dict(ck["opt"])
            done_steps = int(ck["steps"])
            it0 = int(ck["iter"])
            print(f"chay tiep tu {done_steps:,} buoc (vong {it0})")

    b = vram_budget(device)
    print("=" * 70)
    print(f"may      : {device}" + (f"  ({b[1]:.0f} GB VRAM, con {b[0]:.0f} GB)"
                                    if b else ""))
    print(f"moi truong: {n_envs} the gioi song song, moi vong {a.steps} buoc"
          f" = {n_envs * a.steps:,} buoc/vong")
    print(f"bo nao   : {model.n_params():,} tham so"
          + ("  (nua do chinh xac)" if use_amp else ""))
    print(f"anh      : {P.CAM_W}x{P.CAM_H} RGB, servo {int(P.TILT_MIN * 57.3)}"
          f"..{int(P.TILT_MAX * 57.3)} do")
    print(f"bo dem   : {buf.img.numel() / 1024 ** 3:.2f} GB anh (uint8)")
    print(f"dung an toan: tao file {stop_path}  hoac Ctrl-C")
    print("=" * 70, flush=True)

    asked = {"stop": False}

    def _sig(_s, _f):
        if not asked["stop"]:
            asked["stop"] = True
            print("\n>> da nhan Ctrl-C: chay not vong nay roi luu va thoat <<",
                  flush=True)
    signal.signal(signal.SIGINT, _sig)

    t_start = time.time()
    it = it0
    try:
        while done_steps < a.total_steps:
            if asked["stop"] or os.path.exists(stop_path):
                print("dung an toan...")
                break
            it += 1
            env.progress = min(1.0, done_steps / max(1, a.curriculum_steps))
            t0 = time.time()
            last_v = collect(env, model, buf, state, device, amp=use_amp)
            t_col = time.time() - t0
            t0 = time.time()
            st = algo.update(buf, last_v)
            t_upd = time.time() - t0

            done_steps += n_envs * a.steps
            rec = dict(iter=it, steps=done_steps,
                       rew=round(float(buf.rew.mean()), 4),
                       ret=round(float(env.ep_ret.mean()), 2),
                       charged=round(float(env.ep_charged.mean()), 4),
                       wrong=round(float(env.ep_wrong.mean()), 3),
                       batt=round(float(env.batt.mean()), 3),
                       progress=round(env.progress, 3),
                       sps=int(n_envs * a.steps / max(1e-6, t_col + t_upd)),
                       t_col=round(t_col, 2), t_upd=round(t_upd, 2),
                       **{k: round(v, 4) for k, v in st.items()})
            with open(status_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            if it % a.log_every == 0:
                el = time.time() - t_start
                print(f"vong {it:5d}  {done_steps / 1e6:7.2f}M buoc  "
                      f"thuong {rec['rew']:+7.3f}  sac {rec['charged']:.4f}  "
                      f"nham {rec['wrong']:.2f}  "
                      f"{rec['sps']:6d} buoc/s  "
                      f"(ve {t_col:.1f}s / hoc {t_upd:.1f}s)  "
                      f"{el / 3600:.2f}h", flush=True)

            tmp = os.path.join(a.out, "state.tmp.pt")
            torch.save(dict(model=model.state_dict(), opt=algo.opt.state_dict(),
                            steps=done_steps, iter=it,
                            cfg=vars(a)), tmp)
            shutil.move(tmp, os.path.join(a.out, "state.pt"))
    finally:
        tmp = os.path.join(a.out, "state.tmp.pt")
        torch.save(dict(model=model.state_dict(), opt=algo.opt.state_dict(),
                        steps=done_steps, iter=it, cfg=vars(a)), tmp)
        shutil.move(tmp, os.path.join(a.out, "state.pt"))
        print(f"da luu {os.path.join(a.out, 'state.pt')} tai {done_steps:,} buoc")
    return 0


if __name__ == "__main__":
    sys.exit(main())
