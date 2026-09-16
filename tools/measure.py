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
    ap.add_argument("what", choices=("homing", "detector", "speed", "all"))
    a = ap.parse_args(argv)
    if a.what in ("detector", "all"):
        print("== bo do hoc ==")
        measure_detector()
    if a.what in ("speed", "all"):
        print("== toc do ==")
        measure_speed()
    if a.what in ("homing", "all"):
        print("== ve tram sau khi den bao sang ==")
        measure_homing()
    return 0


if __name__ == "__main__":
    sys.exit(main())
