"""Do lai cac con so trong docs/SIM.md, dung mot lenh.

    python -m tools.measure homing     # den bao sang -> ve duoc bao nhieu %
    python -m tools.measure detector   # do chinh xac cua bo do hoc
    python -m tools.measure speed      # bao nhieu micro giay mot buoc

Trong du an nay, muon biet A hay B gay loi thi TAT HAN B ROI DO LAI, dung
ngoi suy luan. File nay de viec do lai cho re.
"""

import argparse
import math
import random
import sys
import time

import numpy as np

from brain.rule_brain import RuleBrain, factory
from sim import dock_detector, params as P
from sim.fleet import FleetSim, LocalBrains
from sim.geometry import wrap_pi_scalar
from sim.lidar import Lidar
from sim.world import make_fleet_map

SEEDS = (3, 7, 11, 19, 23)


def measure_homing(seeds=SEEDS, trials=4, verbose=True):
    """Den bao sac sang len roi, xe co tu ve duoc HOC CUA MINH khong."""
    ok = fail = wrong = 0
    for seed in seeds:
        for trial in range(trials):
            sim = FleetSim(seed=seed, n_robots=5)
            lb = LocalBrains(lambda rid: RuleBrain(rid, 100 * trial + seed),
                             sim.robot_ids)
            t_drop = 25.0 + trial * 30.0
            state = {"done": False}

            def hook(s, state=state, t_drop=t_drop):
                if not state["done"] and s.t > t_drop:
                    for r in s.robots:
                        r.battery = 0.16
                    state["done"] = True

            sim.run(lb, max_seconds=t_drop + 220.0, on_step=hook)
            for r in sim.robots:
                got = r.n_charges >= 1
                ok += got
                fail += not got
                wrong += r.n_wrong_dock
        if verbose:
            print(f"  ...seed {seed} xong (ve duoc {ok}, khong {fail})", flush=True)
    n = ok + fail
    print(f"den bao sang -> tu ve duoc HOC CUA MINH: {ok}/{n} = {100 * ok / n:.0f}%")
    print(f"so lan cam nham hoc cua xe khac: {wrong} ({wrong / n:.1f} lan/xe)")
    return ok / n


def measure_detector(seeds=(3, 5, 8), trials=60):
    """Do hoc chinh xac toi dau, va bao gia bao nhieu."""
    rng = random.Random(4)
    pos_err, ax_err = [], []
    tp = fp = 0
    for seed in seeds:
        w = make_fleet_map(seed)
        for trial in range(trials):
            px = rng.uniform(0.5, 5.9)
            py = rng.uniform(0.5, 4.3)
            th = rng.uniform(-math.pi, math.pi)
            if not w.on_floor(px, py):
                continue
            if min(math.hypot(px - d.x, py - d.y) for d in w.docks) < 0.6:
                continue
            lid = Lidar(seed=trial)
            scan = None
            for i in range(8):
                s = lid.update(P.DT, (px, py, th), (px, py, th),
                               w.static_segments, [], i * P.DT)
                if s is not None:
                    scan = s
            for c in dock_detector.detect(scan):
                cx = px + c.range * math.cos(th + c.bearing)
                cy = py + c.range * math.sin(th + c.bearing)
                err, dock = min(((math.hypot(cx - d.x, cy - d.y), d)
                                 for d in w.docks), key=lambda z: z[0])
                if err < 0.30:
                    tp += 1
                    pos_err.append(err)
                    ax_err.append(abs(math.degrees(
                        wrap_pi_scalar(th + c.axis - dock.theta))))
                elif err > 0.8:
                    fp += 1
    print(f"thay dung {tp}, bao gia that su (cach moi hoc >0,8 m) {fp}"
          f" -> {100 * fp / max(1, tp + fp):.0f}%")
    print(f"lech vi tri trung vi {np.median(pos_err) * 100:.1f} cm, "
          f"lech truc trung vi {np.median(ax_err):.1f} do")


def measure_crn(dirs=4, n_seeds=40, steps=300, robots=2, sigma=0.05,
                seed=7, verbose=True):
    """Chung so ngau nhien giup duoc bao nhieu.

    ES xep hang theo d = f(theta+sigma*eps) - f(theta-sigma*eps).

      cham rieng: Var(d) = Var(f+) + Var(f-)
      cham chung: Var(d) = Var(f+) + Var(f-) - 2*Cov = 2*Var*(1 - rho)

    Nen toan bo cai loi cua chung so ngau nhien nam gon trong MOT con so:
    rho, he so tuong quan giua diem cua hai ca the khi chay tren cung mot
    hat giong. rho = 0,8 thi nhieu con mot nua; rho = 0 thi khong loi gi.

    Do rho truc tiep on dinh hon han do thang Var(d): phuong sai uoc luong
    tu 10 lan lay mau thi ban than no da sai so hon hai lan roi, hai lan do
    ra nguoc nhau la chuyen thuong. Da mat hai lan do vi cho nay.
    """
    from train.es import noise
    from train.policy import GRUPolicy
    from train.rollout import rollout

    pol = GRUPolicy(n_hidden=16, seed=1)
    base = pol.theta.copy()
    rng = random.Random(seed)
    seeds = [rng.randrange(2 ** 30) for _ in range(n_seeds)]

    rhos, sds = [], []
    for i in range(dirs):
        eps = noise(pol.n_params, 500, i)
        cols = []
        for sign in (1.0, -1.0):
            pol.set_theta(base + sign * sigma * eps)
            cols.append([rollout(pol, sd, steps=steps, n_robots=robots,
                                 progress=0.0, collect_obs=False).score
                         for sd in seeds])
        a_ = np.array(cols[0])
        b_ = np.array(cols[1])
        sds.append(0.5 * (a_.std(ddof=1) + b_.std(ddof=1)))
        da = a_ - a_.mean()
        db = b_ - b_.mean()
        den = np.linalg.norm(da) * np.linalg.norm(db)
        rhos.append(float(np.dot(da, db) / den) if den > 1e-9 else 0.0)

    pol.set_theta(base)
    rho = float(np.mean(rhos))
    cut = math.sqrt(max(0.0, 1.0 - rho))
    se = (1.0 - rho * rho) / math.sqrt(max(1, n_seeds - 3))
    if verbose:
        print(f"  tuong quan diem hai ca the tren cung hat giong: "
              f"rho = {rho:+.2f} (+-{se:.2f})")
        print("  tung huong: " + ", ".join(f"{r:+.2f}" for r in rhos))
        print(f"  -> chung hat giong cat nhieu cua d xuong con "
              f"{100 * cut:.0f}%")
        print(f"  do lech chuan diem giua cac hat giong: {np.mean(sds):.1f}"
              f"  ({n_seeds} hat giong, {steps} buoc, {robots} xe)")
    return rho, cut


def measure_speed(seconds=120.0):
    sim = FleetSim(seed=3, n_robots=5)
    lb = LocalBrains(factory(1), sim.robot_ids)
    t0 = time.time()
    rep = sim.run(lb, max_seconds=seconds)
    dt = time.time() - t0
    per = dt / rep.steps
    print(f"{per * 1e6:.0f} us/buoc cho 5 xe = {per / 5 * 1e6:.0f} us/xe")
    print(f"nhanh hon thoi gian that {P.DT / per:.0f} lan")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Do lai cac con so cua mo phong")
    ap.add_argument("what", choices=("homing", "detector", "speed", "crn", "all"))
    a = ap.parse_args(argv)
    if a.what in ("detector", "all"):
        print("== bo do hoc ==")
        measure_detector()
    if a.what in ("speed", "all"):
        print("== toc do ==")
        measure_speed()
    if a.what in ("crn", "all"):
        print("== chung so ngau nhien giup duoc bao nhieu ==")
        measure_crn()
    if a.what in ("homing", "all"):
        print("== ve tram sau khi den bao sang ==")
        measure_homing()
    return 0


if __name__ == "__main__":
    sys.exit(main())
