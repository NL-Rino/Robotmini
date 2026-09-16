"""Chay mo phong doi xe va xem no chay.

    python -m tools.run_fleet                       # chay MAI, Ctrl-C de ngat nao
    python -m tools.run_fleet --seconds 600         # chi de kiem thu
    python -m tools.run_fleet --realtime --view 10  # xem theo thoi gian that
    python -m tools.run_fleet --brain udp           # nao chay o tien trinh khac

Vong lap KHONG co gioi han so buoc va khong co tap. No chi dung khi DUT KET
NOI BO NAO:
  - che do `local`: Ctrl-C ngat ket noi (xe dung banh, vong lap thoat)
  - che do `udp`  : tat tien trinh `python -m link.brain_server` di
"""

import argparse
import math
import signal
import sys

from brain.rule_brain import factory as rule_factory
from sim import params as P
from sim.fleet import FleetSim, LocalBrains
from sim.world import make_fleet_map

BAR = "=" * 72


def render(sim, cols=68, rows=22):
    """Ve mat bang bang ky tu."""
    x0, y0, x1, y1 = sim.world.bounds
    sx = (cols - 1) / max(1e-6, x1 - x0)
    sy = (rows - 1) / max(1e-6, y1 - y0)
    grid = [[" "] * cols for _ in range(rows)]

    def put(wx, wy, ch, over=True):
        cx = int(round((wx - x0) * sx))
        cy = int(round((y1 - wy) * sy))
        if 0 <= cx < cols and 0 <= cy < rows:
            if over or grid[cy][cx] == " ":
                grid[cy][cx] = ch

    seg = sim.world.static_segments
    for i in range(len(seg)):
        ax, ay, bx, by = seg.ax[i], seg.ay[i], seg.bx[i], seg.by[i]
        n = max(2, int(math.hypot(bx - ax, by - ay) * max(sx, sy) * 1.5))
        for k in range(n + 1):
            f = k / n
            put(ax + (bx - ax) * f, ay + (by - ay) * f, ".", over=False)

    for v in sim.world.voids:
        vx = [p[0] for p in v]
        vy = [p[1] for p in v]
        for i in range(24):
            for j in range(10):
                put(min(vx) + (max(vx) - min(vx)) * i / 23.0,
                    min(vy) + (max(vy) - min(vy)) * j / 9.0, "~")

    for d in sim.world.docks:
        put(d.x, d.y, d.name[-1] if d.code is not None else "x")
    for m in sim.world.movers:
        put(m.x, m.y, "o")
    for b in sim.world.beacons:
        if b.on:
            put(b.x, b.y, "*")
    for r in sim.robots:
        # Xe la chu cai, hoc la chu so - nhin vao la biet ngay cai nao la cai
        # nao. Xe A la cua hoc 1, xe B cua hoc 2, ...
        ch = chr(ord("A") + r.id)
        if r.fallen:
            ch = "!"
        elif r.stranded:
            ch = "X"
        elif r.charging:
            ch = "@"
        elif r.in_slot:
            ch = "?"
        put(r.x, r.y, ch)
    return "\n".join("".join(row) for row in grid)


def status(sim):
    out = []
    for r in sim.robots:
        lamp = "NHAP NHAY" if r.low_lamp else "         "
        out.append(f"  xe{r.id} ma{r.code}  pin {r.battery * 100:5.1f}%  {lamp}"
                   f"  sac {r.n_charges:2d}  cam nham {r.n_wrong_dock:2d}"
                   f"  het pin {r.n_flat}  roi {r.n_falls}"
                   f"  {sim.stats(r)['trang_thai']}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Mo phong doi xe tu hanh")
    ap.add_argument("--robots", type=int, default=5)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--decoys", type=int, default=1,
                    help="so hoc moi nhu: giong het hoc that nhung khong phat gi")
    ap.add_argument("--seconds", type=float, default=None,
                    help="gioi han thoi gian mo phong; mac dinh la KHONG co")
    ap.add_argument("--rescue", type=float, default=None,
                    help="sau bao nhieu giay thi nguoi nhat xe chet bo lai vao hoc")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--view", type=int, default=0,
                    help="ve mat bang moi N buoc (0 = khong ve)")
    ap.add_argument("--brain", choices=("local", "udp"), default="local")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9009)
    a = ap.parse_args(argv)

    world = make_fleet_map(a.seed, n_docks=a.robots, n_decoys=a.decoys)
    sim = FleetSim(world, n_robots=a.robots, seed=a.seed,
                   rescue_seconds=a.rescue)

    print(BAR)
    print("ban do: chu so = hoc sac (x = hoc moi nhu), chu cai = xe "
          "(@ dang sac, ? cam nham hoc, X het pin, ! roi), o = nguoi, "
          "* = den goi, ~ = vuc")
    print(f"mat bang {world.name}: {a.robots} xe, "
          f"{len([d for d in world.docks if d.code is not None])} hoc co ma, "
          f"{len([d for d in world.docks if d.code is None])} hoc moi nhu")
    for r in sim.robots:
        print(f"  xe{r.id} mang ma {r.code} -> hoc {sim.home[r.id].name}")
    print("moi xe xuat phat TU TRONG HOC cua no, DUOI o phia trong, mui huong ra.")
    print(f"den bao sac nhap nhay khi pin duoi {P.BATT_LOW * 100:.0f}% "
          f"({P.BATT_LOW / P.BATT_MOVE_DRAIN:.0f} giay chay) - va chi the thoi.")
    print(BAR)

    if a.brain == "udp":
        from link.sim_link import UdpBrainLink
        link = UdpBrainLink(sim.robots, a.host, a.port)
        print(f"cho bo nao o {a.host}:{a.port} - tat no di la mo phong dung")
    else:
        link = LocalBrains(rule_factory(a.seed), sim.robot_ids)
        print("bo nao chay ngay trong tien trinh nay - Ctrl-C de NGAT KET NOI")

        def _stop(_sig, _frm):
            print("\n>> ngat ket noi bo nao <<")
            link.disconnect()
        signal.signal(signal.SIGINT, _stop)
    print(BAR)

    state = {"n": 0}

    def on_step(s):
        state["n"] += 1
        if a.view and state["n"] % a.view == 0:
            print("\x1b[2J\x1b[H" + render(s))
            print(f"t = {s.t:7.1f}s   buoc {s.steps}")
            print(status(s))
            if s.log:
                for ln in s.log[-4:]:
                    print(f"   {ln[0]:7.1f}s xe{ln[1]}: {ln[2]}")

    rep = sim.run(link, max_seconds=a.seconds, on_step=on_step,
                  realtime=a.realtime)
    link.close()

    print(BAR)
    print(f"dung sau {rep.steps} buoc / {rep.sim_seconds:.0f}s mo phong "
          f"({rep.wall_seconds:.0f}s that) - {rep.stop_reason}")
    print(status(sim))
    print(f"\nnhat ky ({len(sim.log)} su kien):")
    for tt, rid, txt in sim.log[-25:]:
        print(f"  {tt:7.1f}s xe{rid}: {txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
