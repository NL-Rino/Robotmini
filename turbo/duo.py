# -*- coding: utf-8 -*-
"""Nuoi bang NHIEU MAY cung mot luc: card lien va CPU chia nhau quan the.

Do tren may cua ban (i5-7200U + HD 620), mot the he 115.200 buoc-xe:

    card lien 1.348 buoc-xe/giay
    CPU       1.921 buoc-xe/giay

Card thua CPU, nhung hai cai CONG LAI thi hon ca hai. ES chia viec duoc
theo ca the: mot the he cham diem P bo trong so doc lap nhau, nen dua 40%
cho card va 60% cho CPU, chay cung luc, roi ghep diem lai la xong.

CAI KHO khong nam o cho chia viec, ma o CHUNG SO NGAU NHIEN: ca P bo trong
so phai chay tren DUNG cung mat bang, cung cho dat xe, cung nhieu cam bien.
Hai may khac nhau ma phai ra cung mot the gioi. Cho nay ban turbo lam duoc
tu dau, nho mot quyet dinh tu lau:

  - `ops.HostRng` sinh so ngau nhien TREN CPU roi chuyen sang may. Cung hat
    giong thi may nao cung ra cung mot chuoi.
  - Cho dat xe (`start_states`) tinh bang Python thuan tu mot hat giong.
  - Mat bang goi thang `sim.world.make_fleet_map`, tat dinh theo hat giong.

Nen chi can dua cung mot hat giong cho hai may la chung thay cung mot the
gioi, va diem cua chung so sanh duoc voi nhau. Co bai kiem thu chan:
chia doi quan the phai ra DUNG diem cua khi chay nguyen mot lo.

Ti le chia TU DIEU CHINH: sau moi the he, may nao chay xong som hon thi the
he sau duoc giao them. Khong phai doan truoc, va no tu bat kip khi may ban
dang ban lam viec khac.
"""

import threading
import time

import torch

from .policy import BatchPolicy
from .rollout import Rollout


class Duo:
    """Mot lo may cung nuoi mot bo nao."""

    def __init__(self, devices, hidden, threads_cpu=None):
        self.devices = list(devices)
        self.hidden = int(hidden)
        # buoc-xe/giay do duoc cua tung may; chua biet thi coi nhu bang nhau
        self.rate = [None] * len(self.devices)
        self.last = [0.0] * len(self.devices)
        if threads_cpu:
            torch.set_num_threads(int(threads_cpu))

    # ------------------------------------------------------------------
    def split(self, n_pop):
        """Chia `n_pop` ca the cho cac may, theo toc do do duoc lan truoc."""
        n = len(self.devices)
        if n == 1:
            return [n_pop]
        w = [r if r else 1.0 for r in self.rate]
        tot = sum(w)
        out = [max(1, int(round(n_pop * x / tot))) for x in w]
        # chinh lai cho du dung n_pop
        while sum(out) > n_pop:
            out[out.index(max(out))] -= 1
        while sum(out) < n_pop:
            out[out.index(min(out))] += 1
        return out

    # ------------------------------------------------------------------
    def run(self, map_seeds, robots, n_pop, crn_seed, progress, theta,
            steps, norm, station_drift=2.5):
        """Cham diem ca quan the. `theta` la (n_pop, n_params) tren CPU.

        Tra ve (diem (n_pop,), (tong_quan_sat, tong_binh_phuong, so_mau),
        thong_ke, phan_chia).
        """
        share = self.split(n_pop)
        out = [None] * len(self.devices)
        err = [None] * len(self.devices)
        lo = [0] * len(self.devices)
        acc = 0
        for i, k in enumerate(share):
            lo[i] = acc
            acc += k

        def worker(i):
            try:
                dev = self.devices[i]
                n = share[i]
                ro = Rollout(map_seeds, robots, n, dev, seed=crn_seed)
                ro.reset(crn_seed, progress, station_drift)
                bp = BatchPolicy(self.hidden, dev)
                bp.set_norm(*norm)
                th = theta[lo[i]:lo[i] + n].to(dev)
                t0 = time.perf_counter()
                sc, (osum, osq, on) = ro.run(bp, th, steps)
                sc = sc.cpu()
                dt = max(1e-6, time.perf_counter() - t0)
                out[i] = (sc, osum.cpu().double(), osq.cpu().double(), on,
                          ro.stats(), n * ro.unit * steps / dt, dt)
            except Exception as e:                      # noqa: BLE001
                err[i] = e

        if len(self.devices) == 1:
            worker(0)
        else:
            ts = [threading.Thread(target=worker, args=(i,), daemon=True)
                  for i in range(len(self.devices))]
            for t in ts:
                t.start()
            for t in ts:
                t.join()

        song = [i for i in range(len(self.devices)) if out[i] is not None]
        if not song:
            raise err[0] if err[0] else RuntimeError("khong may nao chay duoc")
        if len(song) < len(self.devices):
            # Mot may hong: bo no ra, the he sau khong giao nua.
            for i in range(len(self.devices)):
                if out[i] is None:
                    self.rate[i] = 1e-9
            raise err[[i for i in range(len(self.devices))
                       if out[i] is None][0]]

        sc = torch.cat([out[i][0] for i in song])
        osum = sum(out[i][1] for i in song)
        osq = sum(out[i][2] for i in song)
        on = sum(out[i][3] for i in song)
        st = {}
        for k in out[song[0]][4]:
            st[k] = sum(out[i][4][k] * share[i] for i in song) / max(1, n_pop)
        for i in song:
            # trung binh truot, cho khoi nhay theo mot lan do xui
            r = out[i][5]
            self.rate[i] = r if self.rate[i] is None \
                else 0.6 * self.rate[i] + 0.4 * r
            self.last[i] = out[i][6]
        return sc, (osum, osq, on), st, share

    # ------------------------------------------------------------------
    def report(self, share):
        """Mot dong ngan: may nao lam bao nhieu, nhanh cham the nao."""
        parts = []
        for i, d in enumerate(self.devices):
            ten = "CPU" if d.type == "cpu" else str(d)
            r = self.rate[i]
            parts.append(f"{ten} {share[i]} ca the"
                         + (f" {r:,.0f} buoc-xe/giay" if r else ""))
        return " | ".join(parts)
