# -*- coding: utf-8 -*-
"""Vong huan luyen chay theo lo - cai nay moi dung duoc GPU.

    python -m turbo.train --out runs/gpu1 --gens 500
    python -m turbo.train --out runs/gpu1 --resume runs/gpu1
    python -m turbo.train --list-devices

Khac `train/train.py` o DUY NHAT mot cho, nhung la cho quyet dinh: ban kia
cham diem tung ca the mot (nhieu tien trinh, moi tien trinh mot con xe),
ban nay cham diem CA QUAN THE trong mot phep tinh. Voi quan the 64 va 6 con
xe moi ca the thi mot the he la 384 con xe buoc cung luc - du to de GPU co
viec lam.

Moi thu khac deu giu nguyen de hai ban thay the duoc cho nhau:
  - cung khuon file: state.npz / best.npz doc duoc bang `GRUPolicy.load`
  - cung status.jsonl, cung file STOP de dung an toan
  - cung giao trinh nguoc, cung ham phan thuong, cung chung so ngau nhien

Bo nao nuoi o day cam thang vao con robot that duoc, y nhu ban kia.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from train.es import AdamW, centered_ranks, noise
from train.policy import GRUPolicy
from train.train import HOLDOUT, load_state, save_state

from . import device as DEV
from .duo import Duo
from .policy import BatchPolicy
from .rollout import Rollout


def _devices(spec, quiet=False):
    """`spec` co the la mot ten, mot torch.device, hay nhieu ten cach nhau
    bang dau phay ("dml,cpu"). Nhieu ten thi quan the duoc chia cho ca lo."""
    if isinstance(spec, torch.device):
        return [spec]
    if isinstance(spec, (list, tuple)):
        return [d if isinstance(d, torch.device) else DEV.resolve(d)
                for d in spec]
    if spec and "," in str(spec):
        out = []
        for name in str(spec).split(","):
            name = name.strip()
            if not name:
                continue
            try:
                out.append(DEV.resolve(name))
            except Exception:
                if not quiet:
                    print(f"bo qua '{name}': may nay khong co")
        if not out:
            raise SystemExit("khong may nao trong danh sach dung duoc")
        if not quiet:
            print("chay tren: " + ", ".join(str(d) for d in out))
        return out
    return [DEV.pick(spec, quiet=quiet)]


def _status(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _noise_matrix(n_params, gen, n_pairs, device):
    """(2*n_pairs, n_params): nua tren la +eps, nua duoi la -eps.

    Sinh lai tu hat giong dung nhu ban v1 nen hai ban cho ra cung mot chuoi
    nhieu, va mot phien chay do do ban nay so duoc voi ban kia.
    """
    eps = np.stack([noise(n_params, gen, i) for i in range(n_pairs)])
    both = np.concatenate([eps, -eps], axis=0)
    return (torch.as_tensor(both, dtype=torch.float32).to(device),
            torch.as_tensor(eps, dtype=torch.float32).to(device))


def _es_step(theta, eps, sp, sm, sigma, opt):
    """Buoc ES: xep hang giua tren ca hai phia roi di theo huong tang diem."""
    both = np.concatenate([sp, sm])
    r = centered_ranks(both)
    w = r[:len(sp)] - r[len(sp):]
    if not np.any(w):
        return theta, 0.0
    g = (torch.as_tensor(w, dtype=torch.float32,
                         device=eps.device)[:, None] * eps).sum(0)
    g = g.cpu().double().numpy() / (len(sp) * sigma)
    return opt.step(theta, g), float(np.linalg.norm(g))


def train(out="runs/gpu1", hidden=48, pop=64, sigma=0.05, lr=0.02,
          weight_decay=0.005, steps=1200, robots=3, maps=2, gens=1000,
          resume=None, init=None, curriculum_gens=600, eval_every=20,
          device=None, quiet=False, seed=0, threads=None):
    devs = _devices(device, quiet=quiet)
    dev = devs[0]
    os.makedirs(out, exist_ok=True)
    status_path = os.path.join(out, "status.jsonl")
    stop_path = os.path.join(out, "STOP")
    if os.path.exists(stop_path):
        os.remove(stop_path)

    # Ban goc cua trong so van la GRUPolicy cua ban v1: no giu khuon file,
    # bo chuan hoa Welford, va la thu cuoi cung cam vao robot that.
    pol = GRUPolicy(n_hidden=hidden, seed=seed)
    opt = AdamW(pol.n_params, lr=lr, weight_decay=weight_decay)
    gen0, best = 0, -1e18
    cfg = dict(hidden=hidden, pop=pop, sigma=sigma, lr=lr,
               weight_decay=weight_decay, steps=steps, robots=robots,
               maps=maps, curriculum_gens=curriculum_gens, engine="turbo")

    if init and not resume:
        src, _m = GRUPolicy.load(init)
        if src.n_h != hidden:
            raise SystemExit(f"bo nao {init} co {src.n_h} no, khong khop "
                             f"--hidden {hidden}")
        pol.set_theta(src.theta)
        pol.norm.load(*src.norm.state())
        if not quiet:
            print(f"bat dau tu {init}")

    if resume:
        path = resume
        if os.path.isdir(path):
            path = os.path.join(path, "state.npz")
        if os.path.exists(path):
            gen0, best, old = load_state(path, pol, opt)
            if int(old.get("hidden", hidden)) != hidden:
                raise SystemExit(f"bo nao cu co {old['hidden']} no, khong "
                                 f"khop --hidden {hidden}")
            if not quiet:
                print(f"chay tiep tu the he {gen0}, ky luc {best:.1f}")
        elif not quiet:
            print(f"khong thay {path}, bat dau tu dau")

    n_pairs = max(1, pop // 2)
    n_pop = 2 * n_pairs
    bp = BatchPolicy(hidden, dev)
    duo = Duo(devs, hidden, threads_cpu=threads)
    rng = np.random.default_rng(1234 + gen0)
    done = gen0
    hold = None

    if not quiet:
        print(f"quan the {n_pop}, moi ca the {maps * robots} xe "
              f"-> {n_pop * maps * robots} xe buoc cung luc")
        if len(devs) > 1:
            print("chia cho: " + ", ".join(str(d) for d in devs)
                  + "  (ti le tu dieu chinh theo toc do do duoc)")

    try:
        for gen in range(gen0 + 1, gen0 + gens + 1):
            if os.path.exists(stop_path):
                if not quiet:
                    print("thay file STOP -> dung an toan")
                break
            t0 = time.time()
            progress = min(1.0, (gen - 1) / max(1.0, curriculum_gens))

            # CHUNG SO NGAU NHIEN: ca quan the chay tren DUNG mat bang nay,
            # dung cho dat xe nay, dung nhieu cam bien nay.
            map_seeds = [int(rng.integers(0, 2 ** 31 - 1)) for _ in range(maps)]
            crn = int(rng.integers(0, 2 ** 31 - 1))

            both, eps = _noise_matrix(pol.n_params, gen, n_pairs, torch.device("cpu"))
            theta = torch.as_tensor(pol.theta, dtype=torch.float32
                                    )[None, :] + sigma * both
            sc, (osum, osq, on), st, share = duo.run(
                map_seeds, robots, n_pop, crn, progress, theta, steps,
                pol.norm.state()[:2])
            sc = sc.double().numpy()
            sp, sm = sc[:n_pairs], sc[n_pairs:]

            pol.theta, gnorm = _es_step(pol.theta, eps.to("cpu"), sp, sm,
                                        sigma, opt)
            pol._unpack()

            # Bo chuan hoa cap nhat O CUOI the he - trong mot the he ca quan
            # the phai dung chung mot bo, khong thi diem khong so sanh duoc.
            if on > 0:
                bm = (osum / on).numpy()
                bv = np.maximum((osq / on).numpy() - bm * bm, 0.0)
                pol.norm.update(bm, bv, on)
            mean_score = float(sc.mean())
            rec = dict(gen=gen, score=round(mean_score, 2), sigma=sigma,
                       secs=round(time.time() - t0, 1),
                       sat=round(pol.saturation(), 4),
                       progress=round(progress, 3), grad=round(gnorm, 4),
                       charged=round(st["charged"], 3),
                       wrong=round(st["wrong"], 2),
                       falls=round(st["falls"], 2), flats=round(st["flats"], 2),
                       beacons=round(st["beacons"], 2),
                       charges_ok=round(st.get("charges_ok", 0.0), 2),
                       cells=round(st.get("cells", 0.0), 1),
                       complete=round(st.get("complete", 0.0), 2),
                       engine="turbo")
            if len(devs) > 1:
                rec["chia"] = list(share)

            if gen % eval_every == 0 or gen == gen0 + 1:
                if hold is None:
                    hold = Rollout(list(HOLDOUT), robots, 1, dev, seed=90)
                ev = _holdout(hold, bp, pol, steps)
                rec.update({"eval": round(ev["score"], 2),
                            "eval_charged": round(ev["charged"], 3),
                            "eval_falls": round(ev["falls"], 2),
                            "eval_flats": round(ev["flats"], 2),
                            "eval_wrong": round(ev["wrong"], 2),
                            "eval_beacons": round(ev["beacons"], 2),
                            "eval_charges_ok": round(ev["charges_ok"], 2),
                            "eval_complete": round(ev["complete"], 2)})
                if ev["score"] > best:
                    best = ev["score"]
                    pol.save(os.path.join(out, "best.npz"),
                             meta=dict(gen=gen, score=ev["score"]))
                    rec["record"] = True

            if pol.saturation() > 0.25:
                rec["canh_bao"] = "trong so dang phinh - tanh sap bao hoa"

            done = gen
            _status(status_path, rec)
            save_state(out, pol, opt, done, cfg, best)
            if not quiet:
                line = (f"the he {gen:5d}  diem {mean_score:8.1f}  "
                        f"{rec['secs']:5.1f}s  giao trinh {progress:4.2f}  "
                        f"cham {rec['beacons']:.2f}  sac-du {rec['charges_ok']:.2f}  "
                        f"xong {rec['complete']:.2f}  roi {rec['falls']:.2f}  "
                        f"het pin {rec['flats']:.2f}  nham {rec['wrong']:.2f}")
                if "eval" in rec:
                    line += f"  || do rieng {rec['eval']:8.1f}"
                    if rec.get("record"):
                        line += "  <- KY LUC"
                print(line, flush=True)
                if len(devs) > 1 and gen % 10 == 0:
                    print("        " + duo.report(share), flush=True)
    finally:
        save_state(out, pol, opt, done, cfg, best)
    return out


def _holdout(ro, bp, pol, steps):
    """Do tren mat bang XE CHUA TUNG THAY, khong xao trong so."""
    ro.reset(90210, progress=1.0)
    bp.set_norm(*pol.norm.state()[:2])
    theta = torch.as_tensor(pol.theta, dtype=torch.float32
                            ).to(bp.device)[None, :]
    sc, _ = ro.run(bp, theta, steps, collect_obs=False)
    out = ro.stats()
    out["score"] = float(sc.mean())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Nuoi bo nao bang ES, theo lo")
    ap.add_argument("--out", default="runs/gpu1")
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.005)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--robots", type=int, default=3)
    ap.add_argument("--maps", type=int, default=2)
    ap.add_argument("--gens", type=int, default=1000)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init", default=None)
    ap.add_argument("--curriculum-gens", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--threads", type=int, default=None,
                    help="so luong CPU cho phep torch dung")
    DEV.add_argument(ap)
    a = ap.parse_args(argv)
    if getattr(a, "list_devices", False):
        DEV.from_args(a)
    dev = a.device
    train(out=a.out, hidden=a.hidden, pop=a.pop, sigma=a.sigma, lr=a.lr,
          weight_decay=a.weight_decay, steps=a.steps, robots=a.robots,
          maps=a.maps, gens=a.gens, resume=a.resume, init=a.init,
          curriculum_gens=a.curriculum_gens, eval_every=a.eval_every,
          device=dev, quiet=a.quiet, threads=a.threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
