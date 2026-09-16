# -*- coding: utf-8 -*-
"""Xem xe lam gi - va tu lai thu.

    python -m v2.view                 # hoi chon may, mo cua so
    python -m v2.view --brain v2/runs/thu1/state.pt
    python -m v2.view --save anh.png  # khong co man hinh thi xuat ra file

Phim:
    W / S     tien / lui            I / K   ngua / cui camera
    A / D     quay trai / phai      Space   dung banh
    M         doi giua LAI TAY va DE BO NAO LAI
    R         tha lai tu dau        Q       thoat

Cai dang quan tam nhat khi xem: luc pin xuong duoi 15%, xe co tu NGUA CAMERA
LEN doc bang ma tren hoc khong, hay cu the cam bua roi moi biet la nham.
"""

import argparse
import math
import os
import sys

import torch

from . import device as dev_mod
from . import params as P
from .camera import render
from .env import FleetEnv3D
from .policy import ActorCritic

COLOR_NAME = ("do", "xanh duong", "vang", "xanh la")


def load_brain(path, device):
    if not path:
        return None
    ck = torch.load(path, map_location=device, weights_only=False)
    m = ActorCritic().to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


def save_frames(env, path, n=8, big=4):
    """Khong co man hinh: xuat mot bang anh camera ra file PNG."""
    from .tools.png import save_grid
    img = render(env.sc, env.x, env.y, env.th, env.tilt,
                 w=P.CAM_W * big, h=P.CAM_H * big, chunk=env.chunk)
    save_grid(path, img[:n].cpu(), cols=min(4, n), scale=1)
    print(f"da ghi {path}")


class Viewer:
    def __init__(self, env, model, device, idx=0):
        import tkinter as tk
        self.tk = tk
        self.env = env
        self.model = model
        self.device = device
        self.i = idx
        self.manual = model is None
        self.speed = 1
        self.running = True
        self.h = (model.initial_state(env.n, device) if model is not None
                  else None)
        self.keys = set()

        self.root = tk.Tk()
        self.root.title("Robotmini v2 - xe co mat")
        self.root.configure(bg="#12161c")
        top = tk.Frame(self.root, bg="#12161c")
        top.pack(padx=12, pady=10)
        self.cam = tk.Canvas(top, width=P.CAM_W * 6, height=P.CAM_H * 6,
                             bg="black", highlightthickness=0)
        self.cam.grid(row=0, column=0, padx=(0, 12))
        self.map = tk.Canvas(top, width=420, height=P.CAM_H * 6, bg="#0c1016",
                             highlightthickness=0)
        self.map.grid(row=0, column=1)
        self.info = tk.Label(self.root, text="", bg="#12161c", fg="#e6edf3",
                             font=("Consolas", 10), justify="left", anchor="w")
        self.info.pack(fill="x", padx=12, pady=(0, 10))
        tk.Label(self.root, bg="#12161c", fg="#8b98a8", font=("Consolas", 9),
                 text="W/S tien lui   A/D quay   I/K camera len xuong   "
                      "Space dung   M doi lai tay/bo nao   R tha lai   Q thoat"
                 ).pack(pady=(0, 10))

        self.photo = tk.PhotoImage(width=P.CAM_W, height=P.CAM_H)
        self.root.bind("<KeyPress>", self._down)
        self.root.bind("<KeyRelease>", self._up)
        self.root.after(30, self._tick)

    def _down(self, e):
        k = e.keysym.lower()
        self.keys.add(k)
        if k == "q":
            self.root.destroy()
        elif k == "m":
            if self.model is not None:
                self.manual = not self.manual
        elif k == "r":
            self.env.reset_idx(torch.ones(self.env.n, dtype=torch.bool,
                                          device=self.device))
            if self.h is not None:
                self.h = self.model.initial_state(self.env.n, self.device)

    def _up(self, e):
        self.keys.discard(e.keysym.lower())

    def _action(self, img, sca):
        if self.manual or self.model is None:
            a = torch.zeros(self.env.n, 3, device=self.device)
            fwd = (("w" in self.keys) - ("s" in self.keys)) * 0.6
            turn = (("a" in self.keys) - ("d" in self.keys)) * 0.5
            tilt = ("i" in self.keys) - ("k" in self.keys)
            if "space" in self.keys:
                fwd = turn = 0.0
            a[self.i, 0] = fwd - turn
            a[self.i, 1] = fwd + turn
            a[self.i, 2] = tilt
            return a
        with torch.no_grad():
            act, _lp, _v, self.h = self.model.act(img, sca, self.h)
        return act

    def _tick(self):
        if not self.running:
            return
        env = self.env
        img, sca = env.observe()
        a = self._action(img, sca)
        for _ in range(self.speed):
            env.step(a)
        self._draw_cam(img[self.i])
        self._draw_map()
        self._draw_info()
        self.root.after(30, self._tick)

    def _draw_cam(self, img):
        a = (img.clamp(0, 1) * 255).to("cpu").byte().permute(1, 2, 0).numpy()
        rows = " ".join("{" + " ".join("#%02x%02x%02x" % tuple(px)
                                       for px in row) + "}" for row in a)
        self.photo.put(rows, to=(0, 0))
        self.cam.delete("all")
        self.cam.create_image(0, 0, image=self.photo, anchor="nw")
        self.cam.scale("all", 0, 0, 6, 6)
        # phong to bang cach ve lai tren canvas
        self.cam.delete("all")
        big = self.photo.zoom(6, 6)
        self._big = big
        self.cam.create_image(0, 0, image=big, anchor="nw")

    def _draw_map(self):
        env, i = self.env, self.i
        c = self.map
        c.delete("all")
        fl = env.sc["floor"][i].tolist()
        W, H = int(c["width"]), int(c["height"])
        sx = (W - 16) / max(0.1, fl[2] - fl[0])
        sy = (H - 16) / max(0.1, fl[3] - fl[1])
        s = min(sx, sy)

        def pt(x, y):
            return 8 + (x - fl[0]) * s, H - 8 - (y - fl[1]) * s
        for j in range(env.sc["box_c"].shape[1]):
            if not bool(env.sc["box_solid"][i, j]):
                continue
            cx, cy = env.sc["box_c"][i, j, :2].tolist()
            hx, hy = env.sc["box_h"][i, j, :2].tolist()
            yaw = float(env.sc["box_yaw"][i, j])
            pts = []
            for dx, dy in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
                rx = cx + dx * math.cos(yaw) - dy * math.sin(yaw)
                ry = cy + dx * math.sin(yaw) + dy * math.cos(yaw)
                pts += list(pt(rx, ry))
            c.create_polygon(pts, fill="#2a3340", outline="#3d4a5c")
        for j in range(env.sc["cyl_c"].shape[1]):
            r = float(env.sc["cyl_r"][i, j])
            if r < 1e-3:
                continue
            cx, cy = env.sc["cyl_c"][i, j].tolist()
            x0, y0 = pt(cx - r, cy + r)
            x1, y1 = pt(cx + r, cy - r)
            c.create_oval(x0, y0, x1, y1, fill="#4a3f6b", outline="#6b5da0")
        for j in range(env.n_docks):
            dx, dy, dth = env.sc["dock_pose"][i, j].tolist()
            x, y = pt(dx, dy)
            own = j == int(env.home[i])
            col = "#5ddb8f" if own else ("#8b98a8"
                                         if bool(env.sc["dock_powered"][i, j])
                                         else "#6b5a30")
            c.create_oval(x - 5, y - 5, x + 5, y + 5, fill=col, outline="")
            ax, ay = pt(dx + 0.4 * math.cos(dth), dy + 0.4 * math.sin(dth))
            c.create_line(x, y, ax, ay, fill=col)
        rx, ry = pt(float(env.x[i]), float(env.y[i]))
        rr = P.BODY_RADIUS * s
        col = ("#5ddb8f" if bool(env.charging[i]) else
               "#ffb74d" if bool(env.in_slot[i]) else
               "#ff6b6b" if bool(env.stranded[i] or env.fallen[i]) else "#4da3ff")
        c.create_oval(rx - rr, ry - rr, rx + rr, ry + rr, outline=col, width=2)
        th = float(env.th[i])
        hx, hy = pt(float(env.x[i]) + P.BODY_RADIUS * math.cos(th),
                    float(env.y[i]) + P.BODY_RADIUS * math.sin(th))
        c.create_line(rx, ry, hx, hy, fill=col, width=3)
        # huong camera dang nhin
        c.create_line(rx, ry, *pt(float(env.x[i]) + 1.2 * math.cos(th),
                                  float(env.y[i]) + 1.2 * math.sin(th)),
                      fill="#ffffff", dash=(3, 4))

    def _draw_info(self):
        env, i = self.env, self.i
        code = env.sc["dock_code"][i, int(env.home[i])].tolist()
        st = ("DANG SAC" if bool(env.charging[i]) else
              "CAM NHAM HOC - khong ra dien" if bool(env.in_slot[i]) else
              "HET PIN" if bool(env.stranded[i]) else
              "ROI" if bool(env.fallen[i]) else "dang chay")
        lamp = "  [DEN BAO SAC NHAP NHAY]" if bool(env.low_lamp[i]) else ""
        self.info.configure(
            text=(f"{'LAI TAY' if self.manual else 'BO NAO LAI'}   "
                  f"pin {float(env.batt[i]) * 100:5.1f}%{lamp}\n"
                  f"ma hoc cua xe: {' - '.join(COLOR_NAME[c] for c in code)}\n"
                  f"camera nghieng {math.degrees(float(env.tilt[i])):+5.1f} do"
                  f"   |   {st}"))

    def run(self):
        self.root.mainloop()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Xem xe co mat lam gi")
    dev_mod.add_argument(ap)
    ap.add_argument("--brain", default=None, help="duong dan state.pt")
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--save", default=None,
                    help="khong mo cua so, xuat mot bang anh ra file PNG")
    a = ap.parse_args(argv)
    device = dev_mod.from_args(a)
    env = FleetEnv3D(n_envs=a.envs, device=device, seed=a.seed, progress=0.6)
    model = load_brain(a.brain, device) if a.brain else None

    if a.save:
        save_frames(env, a.save, n=a.envs)
        return 0
    try:
        __import__("tkinter")
    except ImportError:
        print("khong co tkinter tren may nay. Dung --save anh.png de xuat "
              "anh ra file thay vi mo cua so.")
        return 1
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        print("khong thay man hinh (DISPLAY rong). Dung --save anh.png.")
        return 1
    Viewer(env, model, device).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
