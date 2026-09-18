# -*- coding: utf-8 -*-
"""Robotmini - phan mem dieu khien.

    python app.py

Ba man hinh:
  CHINH     chon bo nao, xem xe chay, tao bo nao moi, hoac vao huan luyen
  HUAN LUYEN chay ES o mot tien trinh rieng; nut Dung ghi mot file STOP roi
             cho tien trinh do luu xong state.npz moi quay ve
  XEM CHAY   tha xe ra mat bang va nhin no di

O man huan luyen co o "Chay bang": chon giua hai bo may chay khac nhau. Ca
hai nuoi ra DUNG mot loai bo nao va ghi ra dung mot khuon file, chi khac
cach chia viec - xem `turbo/docs/CHAY_BANG_GI.md`. May co card do hoa (ke
ca card lien Intel qua DirectML) thi no hien ra san trong o do.

Dung Tkinter - co san trong Python, khong phai cai gi them, chay duoc tren
Windows cua ban.
"""

import json
import math
import os
import subprocess
import sys
import time
import tkinter as tk
from tkinter import messagebox, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from brain.rule_brain import RuleBrain            # noqa: E402
from sim import params as P                       # noqa: E402
from sim.fleet import FleetSim                    # noqa: E402
from sim.world import make_fleet_map              # noqa: E402
from train.policy import GRUPolicy, PolicyBrain   # noqa: E402
# `turbo/` can PyTorch; ban v1 thi khong. Ai chi muon chay ban v1 thi khong
# phai cai PyTorch, nen cho nay hong thi bo qua chu khong duoc lam chet
# ca phan mem.
try:
    from turbo import device as TDEV           # noqa: E402
except Exception as _e:                        # PyTorch chua cai
    TDEV = None
    _TURBO_WHY = str(_e)

BRAINS = os.path.join(HERE, "brains")
RUNS = os.path.join(HERE, "runs")

BG = "#12161c"
PANEL = "#1b222c"
FG = "#e6edf3"
DIM = "#8b98a8"
ACC = "#4da3ff"
WARN = "#ffb74d"
BAD = "#ff6b6b"
GOOD = "#5ddb8f"
FONT = ("Segoe UI", 10)
FONT_B = ("Segoe UI", 11, "bold")
FONT_H = ("Segoe UI", 16, "bold")
MONO = ("Consolas", 9)


# ---------------------------------------------------------------- tien ich
def list_brains():
    """Tat ca bo nao dung duoc: trong brains/ va best.npz cua cac lan chay."""
    out = [("Bo luat viet tay (khong phai mang)", None)]
    for folder, label in ((BRAINS, ""), ):
        if os.path.isdir(folder):
            for fn in sorted(os.listdir(folder)):
                if fn.endswith(".npz"):
                    out.append((label + fn[:-4], os.path.join(folder, fn)))
    if os.path.isdir(RUNS):
        for run in sorted(os.listdir(RUNS)):
            for which in ("best.npz", "state.npz"):
                path = os.path.join(RUNS, run, which)
                if os.path.exists(path):
                    out.append((f"{run} / {which[:-4]}", path))
    return out


def load_brain_factory(path):
    """Tra ve (ham tao nao, mo ta)."""
    if path is None:
        return (lambda rid: RuleBrain(rid, 1)), "bo luat viet tay"
    if os.path.basename(path) == "state.npz":
        import numpy as np
        d = np.load(path, allow_pickle=True)
        cfg = json.loads(str(d["cfg"]))
        pol = GRUPolicy(n_hidden=int(cfg["hidden"]))
        pol.set_theta(d["theta"])
        pol.norm.load(d["norm_mean"], d["norm_var"], float(d["norm_count"]))
        desc = f"GRU {pol.n_h} no, {pol.n_params} tham so, the he {int(d['gen'])}"
    else:
        pol, meta = GRUPolicy.load(path)
        gen = int(meta.get("gen", 0)) if "gen" in meta else 0
        desc = f"GRU {pol.n_h} no, {pol.n_params} tham so"
        if gen:
            desc += f", the he {gen}"
    return (lambda rid: PolicyBrain(pol, rid)), desc


def _engines():
    """Cac bo may chay co the chon. Khong co PyTorch thi chi con ban v1."""
    if TDEV is not None:
        return TDEV.engines()
    return [("tung xe mot - CPU, nhieu tien trinh", "train.train", None,
             "chua cai PyTorch nen khong co muc chay theo lo")]


def train_command(mod, dev_key, run_dir, hidden, cfg, resume=False,
                  init=None):
    """Dung dong lenh cho tien trinh huan luyen.

    De o day, ngoai lop giao dien, de con kiem thu duoc: may nao khong co
    Tkinter (may chu thue chang han) van goi duoc ham nay.
    """
    args = [sys.executable, "-u", "-m", mod,
            "--out", run_dir, "--hidden", str(hidden),
            "--pop", str(cfg["pop"]), "--steps", str(cfg["steps"]),
            "--robots", str(cfg["robots"]), "--gens", str(cfg["gens"]),
            "--curriculum-gens", str(cfg["curriculum_gens"])]
    if mod == "turbo.train":
        # Ban theo lo khong chia viec cho tien trinh nao ca: ca quan the nam
        # trong mot phep tinh, nen "So luong" khong con nghia gi.
        args += ["--maps", str(cfg["episodes"]), "--device", dev_key or "cpu"]
    else:
        args += ["--episodes", str(cfg["episodes"]),
                 "--jobs", str(cfg["jobs"])]
    if resume:
        args += ["--resume", run_dir]
    elif init:
        args += ["--init", init]
    return args


def button(parent, text, cmd, kind="normal", width=None):
    colors = {"normal": (ACC, "#0b1220"), "stop": (BAD, "#1a0c0c"),
              "ghost": (PANEL, FG), "go": (GOOD, "#08160f")}
    bg, fg = colors[kind]
    b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                  activebackground=bg, activeforeground=fg, font=FONT_B,
                  relief="flat", bd=0, padx=16, pady=9, cursor="hand2")
    if width:
        b.configure(width=width)
    return b


# ---------------------------------------------------------------- ve ban do
class WorldCanvas(tk.Canvas):
    """Ve mat bang va cac xe."""

    def __init__(self, parent, **kw):
        super().__init__(parent, bg="#0c1016", highlightthickness=0, **kw)
        self.sim = None
        self._scale = 1.0
        self._ox = self._oy = 0.0

    def attach(self, sim):
        self.sim = sim
        self.after(30, self._fit)

    def _fit(self):
        if self.sim is None:
            return
        w = max(50, self.winfo_width())
        h = max(50, self.winfo_height())
        x0, y0, x1, y1 = self.sim.world.bounds
        self._scale = min((w - 24) / max(0.1, x1 - x0),
                          (h - 24) / max(0.1, y1 - y0))
        self._ox = 12 - x0 * self._scale
        self._oy = h - 12 + y0 * self._scale

    def _pt(self, x, y):
        return x * self._scale + self._ox, self._oy - y * self._scale

    def redraw(self):
        if self.sim is None:
            return
        self._fit()
        self.delete("all")
        sim = self.sim
        w = sim.world

        for v in w.voids:
            pts = [c for p in v for c in self._pt(*p)]
            self.create_polygon(pts, fill="#000000", outline="#3a2b18", width=2)

        seg = w.static_segments
        for i in range(len(seg)):
            ax, ay = self._pt(seg.ax[i], seg.ay[i])
            bx, by = self._pt(seg.bx[i], seg.by[i])
            self.create_line(ax, ay, bx, by, fill="#3d4a5c", width=2)

        for d in w.docks:
            cx, cy = self._pt(d.x, d.y)
            col = "#2f6b46" if d.code is not None else "#5c4a2a"
            self.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill=col, outline="")
            self.create_text(cx, cy - 15, text=(str(d.code) if d.code else "moi"),
                             fill=DIM, font=("Segoe UI", 8))
            ix, iy = self._pt(d.ir_x, d.ir_y)
            if d.ir_on:
                self.create_oval(ix - 3, iy - 3, ix + 3, iy + 3,
                                 fill="#b03a3a", outline="")

        for b in w.beacons:
            if b.on:
                x, y = self._pt(b.x, b.y)
                self.create_oval(x - 7, y - 7, x + 7, y + 7, outline=WARN, width=2)
                self.create_text(x, y, text="*", fill=WARN, font=FONT_B)

        for m in w.movers:
            x, y = self._pt(m.x, m.y)
            r = m.radius * self._scale
            self.create_oval(x - r, y - r, x + r, y + r, fill="#4a3f6b",
                             outline="#6b5da0")

        for r in sim.robots:
            x, y = self._pt(r.x, r.y)
            rad = P.BODY_RADIUS * self._scale
            if r.fallen:
                col, edge = "#2a2a2a", BAD
            elif r.stranded:
                col, edge = "#3a2020", BAD
            elif r.charging:
                col, edge = "#173a26", GOOD
            elif r.in_slot:
                col, edge = "#3a3418", WARN
            else:
                col, edge = "#17293d", ACC
            self.create_oval(x - rad, y - rad, x + rad, y + rad,
                             fill=col, outline=edge, width=2)
            # mui xe
            hx, hy = self._pt(r.x + P.BODY_RADIUS * math.cos(r.th),
                              r.y + P.BODY_RADIUS * math.sin(r.th))
            self.create_line(x, y, hx, hy, fill=edge, width=3)
            # chan tiep dien o duoi
            tx, ty = self._pt(r.x - P.BODY_RADIUS * math.cos(r.th),
                              r.y - P.BODY_RADIUS * math.sin(r.th))
            self.create_oval(tx - 3, ty - 3, tx + 3, ty + 3,
                             fill=(GOOD if r.charging else
                                   BAD if r.in_slot else "#7a879a"), outline="")
            self.create_text(x, y, text=chr(ord("A") + r.id), fill=FG,
                             font=("Segoe UI", 9, "bold"))


# ---------------------------------------------------------------- app
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Robotmini")
        self.geometry("1100x720")
        self.configure(bg=BG)
        self.minsize(900, 600)
        os.makedirs(BRAINS, exist_ok=True)
        os.makedirs(RUNS, exist_ok=True)
        self.frame = None
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.show(MainScreen)

    def show(self, cls, **kw):
        if self.frame is not None:
            self.frame.destroy_safely()
            self.frame.destroy()
        self.frame = cls(self, **kw)
        self.frame.pack(fill="both", expand=True)

    def _on_close(self):
        if self.frame is not None and not self.frame.can_close():
            return
        if self.frame is not None:
            self.frame.destroy_safely()
        self.destroy()


class Screen(tk.Frame):
    def __init__(self, app, **kw):
        super().__init__(app, bg=BG)
        self.app = app

    def destroy_safely(self):
        pass

    def can_close(self):
        return True


# ---------------------------------------------------------------- man chinh
class MainScreen(Screen):
    def __init__(self, app):
        super().__init__(app)
        tk.Label(self, text="Robotmini", bg=BG, fg=FG, font=("Segoe UI", 26, "bold")
                 ).pack(pady=(28, 2))
        tk.Label(self, text="doi xe tu hanh - mo phong, huan luyen, xem chay",
                 bg=BG, fg=DIM, font=FONT).pack(pady=(0, 20))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=40)

        left = tk.Frame(body, bg=PANEL)
        left.pack(side="left", fill="both", expand=True, padx=(0, 14))
        tk.Label(left, text="Chon bo nao", bg=PANEL, fg=FG, font=FONT_B
                 ).pack(anchor="w", padx=16, pady=(14, 6))
        self.listbox = tk.Listbox(left, bg="#0f141b", fg=FG, font=MONO,
                                  selectbackground=ACC, selectforeground="#0b1220",
                                  relief="flat", highlightthickness=0,
                                  activestyle="none")
        self.listbox.pack(fill="both", expand=True, padx=16, pady=(0, 10))
        self.listbox.bind("<<ListboxSelect>>", self._on_pick)
        self.info = tk.Label(left, text="", bg=PANEL, fg=DIM, font=FONT,
                             anchor="w", justify="left")
        self.info.pack(fill="x", padx=16, pady=(0, 14))

        right = tk.Frame(body, bg=BG, width=280)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        button(right, "Xem xe chay", self._watch, "go").pack(fill="x", pady=6)
        button(right, "Huan luyen", self._train).pack(fill="x", pady=6)
        button(right, "Tao bo nao moi", self._new, "ghost").pack(fill="x", pady=6)
        button(right, "Lam moi danh sach", self._refresh, "ghost").pack(fill="x", pady=6)
        tk.Frame(right, bg=BG, height=20).pack()
        button(right, "Thoat", app._on_close, "ghost").pack(fill="x", pady=6)

        tk.Label(self, bg=BG, fg=DIM, font=("Segoe UI", 9), justify="left",
                 text=("xanh la = dang sac   vang = cam nham hoc   do = het pin / roi\n"
                       "cham o duoi xe la chan tiep dien: phai lui duoi vao hoc moi sac duoc")
                 ).pack(pady=14)
        self._refresh()

    def _refresh(self):
        self.items = list_brains()
        self.listbox.delete(0, "end")
        for name, _p in self.items:
            self.listbox.insert("end", "  " + name)
        self.listbox.selection_set(0)
        self._on_pick()

    def _sel(self):
        s = self.listbox.curselection()
        return self.items[s[0]] if s else self.items[0]

    def _on_pick(self, _e=None):
        name, path = self._sel()
        if path is None:
            self.info.configure(text="Bo luat viet tay: khong phai mang, chi doc "
                                     "48 dau vao nhu bo nao hoc duoc.")
            return
        try:
            _f, desc = load_brain_factory(path)
            self.info.configure(text=f"{desc}\n{path}")
        except Exception as e:
            self.info.configure(text=f"khong doc duoc: {e}")

    def _watch(self):
        name, path = self._sel()
        self.app.show(WatchScreen, brain_name=name, brain_path=path)

    def _train(self):
        name, path = self._sel()
        self.app.show(TrainScreen, brain_name=name, brain_path=path)

    def _new(self):
        NewBrainDialog(self.app, on_done=self._refresh)


class NewBrainDialog(tk.Toplevel):
    def __init__(self, app, on_done=None):
        super().__init__(app)
        self.title("Tao bo nao moi")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.on_done = on_done
        self.transient(app)
        self.grab_set()

        tk.Label(self, text="Bo nao moi", bg=BG, fg=FG, font=FONT_H
                 ).grid(row=0, column=0, columnspan=2, padx=20, pady=(18, 4))
        tk.Label(self, text="khoi tao ngau nhien - chua biet gi ca",
                 bg=BG, fg=DIM, font=FONT).grid(row=1, column=0, columnspan=2,
                                                padx=20, pady=(0, 14))

        tk.Label(self, text="Ten", bg=BG, fg=FG, font=FONT).grid(row=2, column=0,
                                                                 sticky="e", padx=10)
        self.name = tk.Entry(self, bg=PANEL, fg=FG, insertbackground=FG,
                             relief="flat", font=FONT, width=22)
        self.name.insert(0, "nao_" + time.strftime("%m%d_%H%M"))
        self.name.grid(row=2, column=1, sticky="w", pady=4, padx=(0, 20))

        tk.Label(self, text="So no an", bg=BG, fg=FG, font=FONT).grid(row=3, column=0,
                                                                      sticky="e", padx=10)
        self.hidden = ttk.Combobox(self, values=("12", "16", "24", "32"),
                                   width=19, state="readonly")
        self.hidden.set("16")
        self.hidden.grid(row=3, column=1, sticky="w", pady=4, padx=(0, 20))
        self.hidden.bind("<<ComboboxSelected>>", self._count)

        tk.Label(self, text="Hat giong", bg=BG, fg=FG, font=FONT).grid(row=4, column=0,
                                                                       sticky="e", padx=10)
        self.seed = tk.Entry(self, bg=PANEL, fg=FG, insertbackground=FG,
                             relief="flat", font=FONT, width=22)
        self.seed.insert(0, "0")
        self.seed.grid(row=4, column=1, sticky="w", pady=4, padx=(0, 20))

        self.note = tk.Label(self, text="", bg=BG, fg=DIM, font=FONT)
        self.note.grid(row=5, column=0, columnspan=2, pady=(10, 2))
        self._count()

        bar = tk.Frame(self, bg=BG)
        bar.grid(row=6, column=0, columnspan=2, pady=16)
        button(bar, "Tao", self._create, "go").pack(side="left", padx=6)
        button(bar, "Huy", self.destroy, "ghost").pack(side="left", padx=6)

    def _count(self, _e=None):
        h = int(self.hidden.get())
        n = GRUPolicy(n_hidden=h).n_params
        self.note.configure(text=f"GRU 48 -> {h} -> 2, {n} tham so.\n"
                                 f"no cang nhieu cang manh nhung ES cang lau hoi tu.")

    def _create(self):
        name = self.name.get().strip() or "nao_moi"
        try:
            h = int(self.hidden.get())
            seed = int(self.seed.get())
        except ValueError:
            messagebox.showerror("Loi", "So no an va hat giong phai la so nguyen")
            return
        pol = GRUPolicy(n_hidden=h, seed=seed)
        path = os.path.join(BRAINS, name + ".npz")
        pol.save(path, meta=dict(gen=0))
        messagebox.showinfo("Xong", f"Da tao {name}.npz\n\n{pol.n_params} tham so, "
                                    f"chua duoc huan luyen gi ca.")
        if self.on_done:
            self.on_done()
        self.destroy()


# ---------------------------------------------------------------- huan luyen
class TrainScreen(Screen):
    def __init__(self, app, brain_name="", brain_path=None):
        super().__init__(app)
        self.brain_name = brain_name
        self.brain_path = brain_path
        self.proc = None
        self.run_dir = None
        self.status_pos = 0
        self.history = []
        self.eval_hist = []
        self.t_start = 0.0
        self.stopping = False

        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=30, pady=(22, 6))
        tk.Label(head, text="Huan luyen", bg=BG, fg=FG, font=FONT_H).pack(side="left")
        self.badge = tk.Label(head, text="chua chay", bg=BG, fg=DIM, font=FONT_B)
        self.badge.pack(side="right")

        self.setup = tk.Frame(self, bg=PANEL)
        self.setup.pack(fill="x", padx=30, pady=6)
        self._build_setup()

        self.live = tk.Frame(self, bg=BG)
        self._build_live()

        self.bar = tk.Frame(self, bg=BG)
        self.bar.pack(fill="x", padx=30, pady=16)
        self.b_start = button(self.bar, "Bat dau huan luyen", self._start, "go")
        self.b_start.pack(side="left", padx=(0, 10))
        self.b_stop = button(self.bar, "DUNG AN TOAN", self._stop, "stop")
        self.b_back = button(self.bar, "Ve man hinh chinh",
                             lambda: self.app.show(MainScreen), "ghost")
        self.b_back.pack(side="right")

    # ------------------------------------------------------------------
    def _build_setup(self):
        f = self.setup
        hidden = 16
        if self.brain_path and self.brain_path.endswith(".npz"):
            try:
                _fac, desc = load_brain_factory(self.brain_path)
                hidden = int(desc.split("GRU ")[1].split(" no")[0])
            except Exception:
                pass
        src = ("bat dau tu bo nao: " + self.brain_name
               if self.brain_path else "bo luat viet tay khong huan luyen duoc "
                                       "- se tao bo nao moi 16 no")
        tk.Label(f, text=src, bg=PANEL, fg=(FG if self.brain_path else WARN),
                 font=FONT_B).grid(row=0, column=0, columnspan=8, sticky="w",
                                   padx=16, pady=(12, 8))

        # Chay bang gi. Hai bo may chay cho ra cung mot loai bo nao va cung
        # mot khuon file; khac nhau o cach chia viec, va do la tat ca.
        self.engines = _engines()
        er = tk.Frame(f, bg=PANEL)
        er.grid(row=100, column=0, columnspan=8, sticky="w", padx=16, pady=(0, 10))
        tk.Label(er, text="Chay bang", bg=PANEL, fg=FG, font=FONT
                 ).pack(side="left", padx=(0, 8))
        self.engine = tk.StringVar(value=self.engines[0][0])
        cb = ttk.Combobox(er, textvariable=self.engine, state="readonly",
                          width=46, font=FONT,
                          values=[e[0] for e in self.engines])
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", self._on_engine)
        self.engine_tip = tk.Label(er, text="", bg=PANEL, fg=DIM,
                                   font=("Segoe UI", 8))
        self.engine_tip.pack(side="left", padx=10)

        self.vars = {}
        specs = [
            ("pop", "Quan the", 24, "so ca the moi the he (chan, chia doi guong)"),
            ("steps", "So buoc/lan", 400, "moi lan danh gia dai bao nhieu buoc"),
            ("robots", "So xe", 3, "so xe tren mot mat bang khi huan luyen"),
            ("episodes", "So mat bang", 2, "nho cung duoc vi da dung chung hat giong"),
            ("jobs", "So luong", max(1, (os.cpu_count() or 2)),
             "so tien trinh song song (chi dung khi chay tung xe mot)"),
            ("gens", "So the he", 5000, "cu de to, dung luc nao thi bam Dung"),
            ("curriculum_gens", "Giao trinh", 600, "sau bao nhieu the he thi het de"),
        ]
        for i, (key, label, default, tip) in enumerate(specs):
            col = i % 4
            row = 1 + 2 * (i // 4)
            tk.Label(f, text=label, bg=PANEL, fg=FG, font=FONT
                     ).grid(row=row, column=col * 2, sticky="e", padx=(16, 6), pady=3)
            v = tk.StringVar(value=str(default))
            self.vars[key] = v
            tk.Entry(f, textvariable=v, bg="#0f141b", fg=FG, insertbackground=FG,
                     relief="flat", font=FONT, width=9
                     ).grid(row=row, column=col * 2 + 1, sticky="w", pady=3)
            tk.Label(f, text=tip, bg=PANEL, fg=DIM, font=("Segoe UI", 8)
                     ).grid(row=row + 1, column=col * 2, columnspan=2,
                            sticky="w", padx=16)
        self.hidden = hidden
        tk.Label(f, text=f"mang: GRU 48 -> {hidden} -> 2", bg=PANEL, fg=DIM,
                 font=FONT).grid(row=99, column=0, columnspan=8, sticky="w",
                                 padx=16, pady=(6, 4))
        self._on_engine()

    def _on_engine(self, _e=None):
        for label, _mod, _dev, tip in self.engines:
            if label == self.engine.get():
                self.engine_tip.configure(text=tip)
                return

    def _build_live(self):
        top = tk.Frame(self.live, bg=BG)
        top.pack(fill="x", padx=30)
        self.tiles = {}
        for key, label in (("gen", "The he"), ("score", "Diem quan the"),
                           ("eval", "Diem do rieng"), ("best", "Ky luc"),
                           ("secs", "Giay/the he"), ("eta", "Da chay")):
            box = tk.Frame(top, bg=PANEL)
            box.pack(side="left", fill="both", expand=True, padx=4)
            tk.Label(box, text=label, bg=PANEL, fg=DIM, font=("Segoe UI", 9)
                     ).pack(pady=(10, 0))
            lab = tk.Label(box, text="-", bg=PANEL, fg=FG, font=("Segoe UI", 17, "bold"))
            lab.pack(pady=(0, 10))
            self.tiles[key] = lab

        mid = tk.Frame(self.live, bg=BG)
        mid.pack(fill="both", expand=True, padx=30, pady=10)
        self.plot = tk.Canvas(mid, bg="#0c1016", highlightthickness=0, height=200)
        self.plot.pack(fill="both", expand=True)
        self.log = tk.Text(mid, bg="#0c1016", fg=DIM, font=MONO, height=9,
                           relief="flat", highlightthickness=0, wrap="none")
        self.log.pack(fill="x", pady=(10, 0))

    # ------------------------------------------------------------------
    def _start(self):
        try:
            cfg = {k: int(v.get()) for k, v in self.vars.items()}
        except ValueError:
            messagebox.showerror("Loi", "Cac o cai dat phai la so nguyen")
            return
        if cfg["pop"] < 4:
            messagebox.showerror("Loi", "Quan the phai it nhat 4")
            return

        # Ten thu muc chay. Neu bo nao duoc chon von da nam trong runs/ thi
        # lay ten LAN CHAY do, khong lay ten file - chon "demo / state" ma
        # tao ra runs/state/ thi lan sau khong ai tim ra no o dau.
        if self.brain_path:
            folder = os.path.dirname(os.path.abspath(self.brain_path))
            if os.path.dirname(folder) == os.path.abspath(RUNS):
                name = os.path.basename(folder)
            else:
                name = os.path.basename(self.brain_path)[:-4]
        else:
            name = "nao_moi"
        self.run_dir = os.path.join(RUNS, name)
        os.makedirs(self.run_dir, exist_ok=True)
        stop_file = os.path.join(self.run_dir, "STOP")
        if os.path.exists(stop_file):
            os.remove(stop_file)

        state = os.path.join(self.run_dir, "state.npz")
        mod, dev_key = "train.train", None
        for label, m, d, _tip in self.engines:
            if label == self.engine.get():
                mod, dev_key = m, d
        resume = False
        if os.path.exists(state):
            resume = messagebox.askyesno(
                "Chay tiep?", f"Da co {state}.\n\nChay tiep tu do (Co) "
                              f"hay bat dau lai tu dau (Khong)?")
        args = train_command(mod, dev_key, self.run_dir, self.hidden, cfg,
                             resume=resume,
                             init=None if resume else self.brain_path)

        self.status_pos = 0
        self.history = []
        self.eval_hist = []
        kw = {}
        if os.name == "nt":
            kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(args, cwd=HERE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.STDOUT, **kw)
        self.t_start = time.time()

        self.setup.pack_forget()
        self.live.pack(fill="both", expand=True)
        self.b_start.pack_forget()
        self.b_back.pack_forget()
        self.b_stop.pack(side="left")
        self.badge.configure(text="DANG HUAN LUYEN", fg=GOOD)
        self._log(f"chay: {' '.join(args[2:])}")
        self.after(400, self._poll)

    def _log(self, line):
        self.log.insert("end", line + "\n")
        self.log.see("end")
        if int(self.log.index("end-1c").split(".")[0]) > 300:
            self.log.delete("1.0", "50.0")

    def _poll(self):
        if self.proc is None:
            return
        self._read_status()
        alive = self.proc.poll() is None
        if not alive:
            self.after(300, self._read_status)
            if self.stopping:
                self.badge.configure(text="DA DUNG AN TOAN - da luu state.npz",
                                     fg=ACC)
            else:
                self.badge.configure(text="tien trinh huan luyen da ket thuc", fg=WARN)
            self.b_stop.pack_forget()
            self.b_back.pack(side="right")
            self.proc = None
            return
        self.tiles["eta"].configure(
            text=time.strftime("%H:%M:%S", time.gmtime(time.time() - self.t_start)))
        self.after(400, self._poll)

    def _read_status(self):
        path = os.path.join(self.run_dir, "status.jsonl")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                f.seek(self.status_pos)
                lines = f.readlines()
                self.status_pos = f.tell()
        except OSError:
            return
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            self.history.append(rec)
            self.tiles["gen"].configure(text=str(rec["gen"]))
            self.tiles["score"].configure(text=f"{rec['score']:.1f}")
            self.tiles["secs"].configure(text=f"{rec['secs']:.1f}")
            msg = (f"the he {rec['gen']:5d}  diem {rec['score']:8.1f}  "
                   f"giao trinh {rec['progress']:.2f}  sac {rec['charged']:.2f}  "
                   f"roi {rec['falls']}  het pin {rec['flats']}  "
                   f"nham {rec['wrong']}")
            if "eval" in rec:
                self.tiles["eval"].configure(text=f"{rec['eval']:.1f}")
                self.eval_hist.append((rec["gen"], rec["eval"]))
                msg += f"  || do rieng {rec['eval']:.1f}"
                if rec.get("record"):
                    msg += "  <- KY LUC"
                    self.tiles["best"].configure(text=f"{rec['eval']:.1f}", fg=GOOD)
            if "canh_bao" in rec:
                msg += "   !! " + rec["canh_bao"]
            self._log(msg)
        self._draw_plot()

    def _draw_plot(self):
        c = self.plot
        c.delete("all")
        if len(self.history) < 2:
            return
        w = max(50, c.winfo_width())
        h = max(50, c.winfo_height())
        xs = [r["gen"] for r in self.history]
        ys = [r["score"] for r in self.history]
        lo, hi = min(ys), max(ys)
        if self.eval_hist:
            lo = min(lo, min(v for _g, v in self.eval_hist))
            hi = max(hi, max(v for _g, v in self.eval_hist))
        if hi - lo < 1e-6:
            hi = lo + 1.0
        pad = 0.08 * (hi - lo)
        lo, hi = lo - pad, hi + pad

        def pt(g, v):
            x = 30 + (w - 45) * (g - xs[0]) / max(1, xs[-1] - xs[0])
            y = h - 22 - (h - 40) * (v - lo) / (hi - lo)
            return x, y

        if lo < 0 < hi:
            _zx, zy = pt(xs[0], 0.0)
            c.create_line(30, zy, w - 12, zy, fill="#2a3542", dash=(3, 3))
        c.create_line(*[u for g, v in zip(xs, ys) for u in pt(g, v)],
                      fill="#3f6fa0", width=1)
        if len(self.eval_hist) >= 2:
            c.create_line(*[u for g, v in self.eval_hist for u in pt(g, v)],
                          fill=GOOD, width=2)
        c.create_text(34, 12, text=f"{hi:.0f}", fill=DIM, anchor="w", font=("Segoe UI", 8))
        c.create_text(34, h - 12, text=f"{lo:.0f}", fill=DIM, anchor="w",
                      font=("Segoe UI", 8))
        c.create_text(w - 14, 12, text="xanh duong = quan the   xanh la = do rieng",
                      fill=DIM, anchor="e", font=("Segoe UI", 8))

    # ------------------------------------------------------------------
    def _stop(self):
        if self.proc is None:
            return
        self.stopping = True
        self.badge.configure(text="DANG DUNG - cho luu xong state.npz...", fg=WARN)
        self.b_stop.configure(state="disabled", text="dang dung...")
        try:
            open(os.path.join(self.run_dir, "STOP"), "w").close()
        except OSError:
            pass
        self._log(">> da ghi file STOP, cho the he hien tai chay not roi luu <<")
        self.after(300, self._wait_stop, time.time())

    def _wait_stop(self, t0):
        if self.proc is None or self.proc.poll() is not None:
            self.proc = None
            self._read_status()
            self.badge.configure(text="DA DUNG AN TOAN - da luu state.npz", fg=ACC)
            self.b_stop.pack_forget()
            self.b_back.pack(side="right")
            # Doc lai mot lan nua sau mot nhip: dong trang thai cuoi cung co
            # the vua duoc ghi ra ngay truoc luc tien trinh thoat.
            self.after(300, self._read_status)
            return
        waited = time.time() - t0
        self.badge.configure(text=f"DANG DUNG - cho {waited:.0f}s "
                                  f"(het the he nay la luu)", fg=WARN)
        if waited > 600:
            self._log("!! qua 10 phut chua dung, buoc ket thuc tien trinh")
            self.proc.kill()
        self.after(300, self._wait_stop, t0)

    def destroy_safely(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                open(os.path.join(self.run_dir, "STOP"), "w").close()
            except OSError:
                pass

    def can_close(self):
        if self.proc is not None and self.proc.poll() is None:
            return messagebox.askyesno(
                "Dang huan luyen",
                "Dang huan luyen. Dong bay gio se ghi file STOP nhung khong "
                "cho tien trinh luu xong.\n\nNen bam DUNG AN TOAN truoc.\n\n"
                "Van dong?")
        return True


# ---------------------------------------------------------------- xem chay
class WatchScreen(Screen):
    def __init__(self, app, brain_name="", brain_path=None):
        super().__init__(app)
        self.brain_name = brain_name
        self.brain_path = brain_path
        self.running = False
        self.sim = None
        self.brains = {}
        self.speed = 1
        self._last = 0.0

        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=24, pady=(18, 4))
        tk.Label(head, text="Xem xe chay", bg=BG, fg=FG, font=FONT_H).pack(side="left")
        self.sub = tk.Label(head, text=brain_name, bg=BG, fg=ACC, font=FONT_B)
        self.sub.pack(side="left", padx=14)
        self.clock = tk.Label(head, text="", bg=BG, fg=DIM, font=FONT_B)
        self.clock.pack(side="right")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=24, pady=6)
        self.canvas = WorldCanvas(body)
        self.canvas.pack(side="left", fill="both", expand=True)

        side = tk.Frame(body, bg=PANEL, width=310)
        side.pack(side="right", fill="y", padx=(14, 0))
        side.pack_propagate(False)
        tk.Label(side, text="Cai dat", bg=PANEL, fg=FG, font=FONT_B
                 ).pack(anchor="w", padx=14, pady=(12, 6))
        row = tk.Frame(side, bg=PANEL)
        row.pack(fill="x", padx=14)
        tk.Label(row, text="Mat bang", bg=PANEL, fg=FG, font=FONT).pack(side="left")
        self.seed = tk.StringVar(value="3")
        tk.Entry(row, textvariable=self.seed, bg="#0f141b", fg=FG, width=6,
                 relief="flat", insertbackground=FG, font=FONT).pack(side="left", padx=8)
        tk.Label(row, text="So xe", bg=PANEL, fg=FG, font=FONT).pack(side="left")
        self.nrob = tk.StringVar(value="5")
        tk.Entry(row, textvariable=self.nrob, bg="#0f141b", fg=FG, width=4,
                 relief="flat", insertbackground=FG, font=FONT).pack(side="left", padx=8)

        row2 = tk.Frame(side, bg=PANEL)
        row2.pack(fill="x", padx=14, pady=8)
        tk.Label(row2, text="Toc do", bg=PANEL, fg=FG, font=FONT).pack(side="left")
        for mult in (1, 2, 5, 20):
            tk.Button(row2, text=f"x{mult}", font=("Segoe UI", 9), relief="flat",
                      bg="#0f141b", fg=FG, bd=0, padx=8, cursor="hand2",
                      command=lambda m=mult: self._set_speed(m)
                      ).pack(side="left", padx=3)
        self.speed_lab = tk.Label(side, text="x1", bg=PANEL, fg=DIM, font=FONT)
        self.speed_lab.pack(anchor="w", padx=14)

        tk.Label(side, text="Cac xe", bg=PANEL, fg=FG, font=FONT_B
                 ).pack(anchor="w", padx=14, pady=(14, 4))
        self.rows = tk.Frame(side, bg=PANEL)
        self.rows.pack(fill="x", padx=14)
        self.events = tk.Text(side, bg="#0f141b", fg=DIM, font=MONO, height=10,
                              relief="flat", highlightthickness=0, wrap="none")
        self.events.pack(fill="both", expand=True, padx=14, pady=12)

        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=24, pady=14)
        self.b_go = button(bar, "Tha xe ra", self._start, "go")
        self.b_go.pack(side="left")
        self.b_pause = button(bar, "Tam dung", self._toggle, "ghost")
        button(bar, "Ve man hinh chinh", self._back, "ghost").pack(side="right")

        self._robot_labels = []
        self._log_at = 0

    def _set_speed(self, m):
        self.speed = m
        self.speed_lab.configure(text=f"x{m}"
                                 + ("  (nhanh hon thuc te)" if m > 1 else ""))

    def _start(self):
        try:
            seed = int(self.seed.get())
            n = max(1, min(5, int(self.nrob.get())))
        except ValueError:
            messagebox.showerror("Loi", "Mat bang va so xe phai la so nguyen")
            return
        try:
            factory, desc = load_brain_factory(self.brain_path)
        except Exception as e:
            messagebox.showerror("Loi", f"Khong doc duoc bo nao:\n{e}")
            return
        self.sub.configure(text=f"{self.brain_name}  -  {desc}")

        world = make_fleet_map(seed, n_docks=n, n_decoys=1)
        self.sim = FleetSim(world, n_robots=n, seed=seed)
        self.brains = {r.id: factory(r.id) for r in self.sim.robots}
        self.canvas.attach(self.sim)
        self._log_at = 0
        self.events.delete("1.0", "end")

        for w in self._robot_labels:
            w.destroy()
        self._robot_labels = []
        for r in self.sim.robots:
            lab = tk.Label(self.rows, text="", bg=PANEL, fg=FG, font=MONO,
                           anchor="w", justify="left")
            lab.pack(fill="x")
            self._robot_labels.append(lab)

        self.running = True
        self.b_go.configure(text="Tha lai tu dau")
        self.b_pause.pack(side="left", padx=10)
        self._last = time.time()
        self._tick()

    def _toggle(self):
        self.running = not self.running
        self.b_pause.configure(text="Chay tiep" if not self.running else "Tam dung")
        if self.running:
            self._last = time.time()
            self._tick()

    def advance(self):
        """Chay may buoc mo phong roi ve lai. Khong dat lich."""
        if self.sim is None:
            return
        for _ in range(self.speed):
            obs = self.sim.observe()
            cmds = {}
            for r in self.sim.robots:
                cmds[r.id] = self.brains[r.id](obs[r.id], self.sim.t)
            self.sim.step(cmds)
        self.canvas.redraw()
        self._update_side()
        self.clock.configure(text=f"t = {self.sim.t:7.1f}s")

    def _tick(self):
        # Tach lam hai: `advance` chay mo phong, `_tick` dat lich. Gop lam mot
        # thi khong the goi tay mot buoc duoc nua - moi lan goi lai dat them
        # mot lich, va update() cua Tk se chay het cac lich da toi han, moi
        # cai lai de them mot cai: vong lap khong bao gio thoat.
        if not self.running or self.sim is None:
            return
        self.advance()
        dt_ms = max(5, int(P.DT * 1000) - int((time.time() - self._last) * 1000))
        self._last = time.time()
        self.after(dt_ms if self.speed == 1 else 5, self._tick)

    def _update_side(self):
        for lab, r in zip(self._robot_labels, self.sim.robots):
            lamp = "@@" if (r.low_lamp and
                            r.low_battery_blink(self.sim.t) > 0.5) else "  "
            if r.fallen:
                st, col = "roi", BAD
            elif r.stranded:
                st, col = "het pin", BAD
            elif r.charging:
                st, col = "dang sac", GOOD
            elif r.in_slot:
                st, col = "cam NHAM hoc", WARN
            else:
                st, col = "dang chay", FG
            bars = int(round(r.battery * 12))
            lab.configure(
                text=(f"{chr(ord('A') + r.id)} ma{r.code} "
                      f"[{'#' * bars}{'.' * (12 - bars)}] {r.battery * 100:3.0f}% "
                      f"{lamp} {st}  sac{r.n_charges} nham{r.n_wrong_dock}"),
                fg=col)
        while self._log_at < len(self.sim.log):
            t, rid, txt = self.sim.log[self._log_at]
            self._log_at += 1
            self.events.insert("end", f"{t:7.1f}s xe{chr(ord('A') + rid)}: {txt}\n")
            self.events.see("end")

    def _back(self):
        self.running = False
        self.app.show(MainScreen)

    def destroy_safely(self):
        self.running = False


if __name__ == "__main__":
    App().mainloop()
