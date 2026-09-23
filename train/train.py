"""Vong huan luyen.

    python -m train.train --out runs/thu1 --gens 500
    python -m train.train --out runs/thu1 --resume runs/thu1   # chay tiep

Dung an toan: tao mot file ten STOP trong thu muc --out. Vong lap dang chay
the he nao thi chay not the he do, luu state.npz, roi thoat. Giao dien
app.py bam nut Dung la lam dung viec nay.

    state.npz  chay tiep duoc: co trong so, momen cua AdamW, bo chuan hoa,
               so the he, tien do giao trinh
    best.npz   CHI la ban sao luc pha ky luc, KHONG chay tiep duoc

Ban cu vap dung cho nay (loi k): `best.npz` chi duoc ghi khi pha ky luc, nen
den the he 6.000 no van con ghi gen=1744, va "chay tiep" tu no la quay
nguoc ve qua khu.
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

from .es import AdamW, gradient, noise
from .policy import GRUPolicy
from .rollout import evaluate, rollout

HOLDOUT = (90001, 90002, 90003, 90004, 90005, 90006)

_W = {}


def _init_worker(hidden, n_in, n_out):
    _W["pol"] = GRUPolicy(n_hidden=hidden, n_in=n_in, n_out=n_out, seed=0)


def _eval_one(task):
    (theta, gen, idx, sign, sigma, seeds, steps, n_robots, progress,
     nmean, nvar, ncount) = task
    pol = _W["pol"]
    eps = noise(pol.n_params, gen, idx)
    pol.set_theta(np.asarray(theta) + sign * sigma * eps)
    pol.norm.load(nmean, nvar, ncount)

    total = 0.0
    osum = np.zeros(pol.n_in)
    osq = np.zeros(pol.n_in)
    on = 0
    st = dict(charged=0.0, wrong=0, falls=0, flats=0, beacons=0,
              charges_ok=0, cells=0, complete=0)
    for s in seeds:
        r = rollout(pol, s, steps=steps, n_robots=n_robots, progress=progress)
        total += r.score
        osum += r.obs_sum
        osq += r.obs_sqsum
        on += r.obs_n
        st["charged"] += r.charged
        st["wrong"] += r.wrong
        st["falls"] += r.falls
        st["flats"] += r.flats
        st["beacons"] += r.beacons
        st["charges_ok"] += r.charges_ok
        st["cells"] += r.cells
        st["complete"] += r.complete
    return idx, sign, total / len(seeds), osum, osq, on, st


def _status(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_state(out, pol, opt, gen, cfg, best):
    mean, var, count = pol.norm.state()
    tmp = os.path.join(out, "state.tmp.npz")
    np.savez(tmp, theta=pol.theta, gen=gen, best=best,
             norm_mean=mean, norm_var=var, norm_count=count,
             cfg=json.dumps(cfg), **{"opt_" + k: v for k, v in opt.state().items()})
    shutil.move(tmp, os.path.join(out, "state.npz"))


def load_state(path, pol, opt):
    d = np.load(path, allow_pickle=True)
    pol.set_theta(d["theta"])
    pol.norm.load(d["norm_mean"], d["norm_var"], float(d["norm_count"]))
    opt.load({k[4:]: d[k] for k in d.files if k.startswith("opt_")})
    return int(d["gen"]), float(d["best"]), json.loads(str(d["cfg"]))


def train(out="runs/thu1", hidden=48, pop=32, sigma=0.05, lr=0.02,
          weight_decay=0.005, steps=1200, robots=3, episodes=2, gens=1000,
          jobs=0, resume=None, curriculum_gens=600, eval_every=20,
          quiet=False, seed=0, init=None):
    os.makedirs(out, exist_ok=True)
    status_path = os.path.join(out, "status.jsonl")
    stop_path = os.path.join(out, "STOP")
    if os.path.exists(stop_path):
        os.remove(stop_path)

    pol = GRUPolicy(n_hidden=hidden, seed=seed)
    opt = AdamW(pol.n_params, lr=lr, weight_decay=weight_decay)
    gen0, best = 0, -1e18
    cfg = dict(hidden=hidden, pop=pop, sigma=sigma, lr=lr,
               weight_decay=weight_decay, steps=steps, robots=robots,
               episodes=episodes, curriculum_gens=curriculum_gens)

    if init and not resume:
        # Bat dau tu mot bo nao co san (vua tao, hoac da nuoi doi chut).
        # Khac `resume`: khong keo theo momen AdamW hay so the he cu.
        src, _meta = GRUPolicy.load(init)
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
                raise SystemExit(
                    f"bo nao cu co {old['hidden']} no, khong khop --hidden {hidden}")
            cfg.update({k: v for k, v in old.items() if k == "hidden"})
            if not quiet:
                print(f"chay tiep tu the he {gen0}, ky luc {best:.1f}")
        elif not quiet:
            print(f"khong thay {path}, bat dau tu dau")

    n_pairs = max(1, pop // 2)
    rng = np.random.default_rng(1234 + gen0)
    done = gen0      # the he CUOI CUNG DA CHAY XONG, khong phai the he dang do

    pool = None
    if jobs and jobs > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(jobs, initializer=_init_worker,
                        initargs=(hidden, pol.n_in, pol.n_out))
    else:
        _init_worker(hidden, pol.n_in, pol.n_out)

    try:
        for gen in range(gen0 + 1, gen0 + gens + 1):
            if os.path.exists(stop_path):
                if not quiet:
                    print("thay file STOP -> dung an toan")
                break
            t0 = time.time()
            progress = min(1.0, (gen - 1) / max(1.0, curriculum_gens))

            # CHUNG SO NGAU NHIEN: ca quan the dung DUNG bo hat giong nay.
            seeds = [int(rng.integers(0, 2 ** 31 - 1)) for _ in range(episodes)]
            nmean, nvar, ncount = pol.norm.state()
            tasks = []
            for i in range(n_pairs):
                for sign in (1.0, -1.0):
                    tasks.append((pol.theta, gen, i, sign, sigma, seeds, steps,
                                  robots, progress, nmean, nvar, ncount))

            if pool is not None:
                results = pool.map(_eval_one, tasks, chunksize=1)
            else:
                results = [_eval_one(t) for t in tasks]

            sp = [0.0] * n_pairs
            sm = [0.0] * n_pairs
            osum = np.zeros(pol.n_in)
            osq = np.zeros(pol.n_in)
            on = 0
            agg = dict(charged=0.0, wrong=0, falls=0, flats=0, beacons=0,
                       charges_ok=0, cells=0, complete=0)
            for idx, sign, sc, s1, s2, n_obs, st in results:
                (sp if sign > 0 else sm)[idx] = sc
                osum += s1
                osq += s2
                on += n_obs
                for k in agg:
                    agg[k] += st[k]

            g = gradient(pol.n_params, gen, sp, sm, sigma)
            pol.set_theta(opt.step(pol.theta, g))

            # Cap nhat bo chuan hoa O CUOI the he: trong mot the he moi ca
            # the phai dung chung mot bo, khong thi diem khong so sanh duoc.
            if on > 0:
                bm = osum / on
                bv = np.maximum(osq / on - bm * bm, 0.0)
                pol.norm.update(bm, bv, on)

            mean_score = 0.5 * (float(np.mean(sp)) + float(np.mean(sm)))
            rec = dict(gen=gen, score=round(mean_score, 2),
                       sigma=sigma, secs=round(time.time() - t0, 1),
                       sat=round(pol.saturation(), 4),
                       progress=round(progress, 3),
                       grad=round(float(np.linalg.norm(g)), 4),
                       charged=round(agg["charged"], 3), wrong=agg["wrong"],
                       falls=agg["falls"], flats=agg["flats"],
                       beacons=agg["beacons"], charges_ok=agg["charges_ok"],
                       cells=agg["cells"], complete=agg["complete"])

            if gen % eval_every == 0 or gen == gen0 + 1:
                ev = evaluate(pol, HOLDOUT, steps=steps, n_robots=robots,
                              progress=1.0)
                rec["eval"] = round(ev.score, 2)
                rec["eval_charged"] = round(ev.charged, 3)
                rec["eval_falls"] = ev.falls
                rec["eval_flats"] = ev.flats
                rec["eval_wrong"] = ev.wrong
                rec["eval_beacons"] = ev.beacons
                rec["eval_charges_ok"] = ev.charges_ok
                rec["eval_complete"] = ev.complete
                if ev.score > best:
                    best = ev.score
                    pol.save(os.path.join(out, "best.npz"),
                             meta=dict(gen=gen, score=ev.score))
                    rec["record"] = True

            if pol.saturation() > 0.25:
                rec["canh_bao"] = "trong so dang phinh - tanh sap bao hoa"

            done = gen
            _status(status_path, rec)
            save_state(out, pol, opt, done, cfg, best)
            if not quiet:
                line = (f"the he {gen:5d}  diem {mean_score:8.1f}  "
                        f"{rec['secs']:5.1f}s  giao trinh {progress:4.2f}  "
                        f"cham {agg['beacons']}  sac-du {agg['charges_ok']}  "
                        f"xong {agg['complete']}  roi {agg['falls']}  "
                        f"het pin {agg['flats']}  nham {agg['wrong']}")
                if "eval" in rec:
                    line += f"  || do rieng {rec['eval']:8.1f}"
                    if rec.get("record"):
                        line += "  <- KY LUC"
                print(line, flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        # Luu theo `done`: neu thoat giua chung thi the he dang do chua co
        # ket qua, ghi so cua no vao la lan sau chay tiep se nhay coc.
        save_state(out, pol, opt, done, cfg, best)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Nuoi bo nao bang ES")
    ap.add_argument("--out", default="runs/thu1")
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.005)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--robots", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--gens", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init", default=None,
                    help="bat dau tu mot file bo nao co san")
    ap.add_argument("--curriculum-gens", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    train(out=a.out, hidden=a.hidden, pop=a.pop, sigma=a.sigma, lr=a.lr,
          weight_decay=a.weight_decay, steps=a.steps, robots=a.robots,
          episodes=a.episodes, gens=a.gens, jobs=a.jobs, resume=a.resume,
          curriculum_gens=a.curriculum_gens, eval_every=a.eval_every,
          quiet=a.quiet, init=a.init)
    return 0


if __name__ == "__main__":
    sys.exit(main())
