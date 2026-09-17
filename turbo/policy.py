# -*- coding: utf-8 -*-
"""GRU cua ban v1, chay CA QUAN THE cung mot luc.

Ban `train/policy.py` chay MOT bo trong so cho MOT con xe: mot buoc la 11
phep nhan ma tran 16x48. O kich thuoc do thi tien goi ham dat hon tien tinh
toan, va GPU nam khong. O day ta gop lai:

    (P bo trong so) x (G con xe moi bo) x (11 phep nhan)

thanh 11 phep nhan ma tran theo lo. P=64, G=9 thi mot buoc la 11 phep bmm
tren (64, 9, 48) - bay gio moi co viec cho GPU lam.

Khuon tham so giu Y HET ban v1: cung thu tu, cung hinh dang, cung mot vector
phang. Nen file .npz cua ban nay mo duoc bang `GRUPolicy.load` va cam thang
vao con robot that, va nguoc lai.
"""

import numpy as np
import torch

from train.policy import GRUPolicy, N_IN, N_OUT


def shapes(n_h, n_in=N_IN, n_out=N_OUT):
    h, i, o = int(n_h), int(n_in), int(n_out)
    return [("Wz", (h, i)), ("Wr", (h, i)), ("Wn", (h, i)),
            ("Uz", (h, h)), ("Ur", (h, h)), ("Un", (h, h)),
            ("bz", (h,)), ("br", (h,)), ("bn", (h,)),
            ("Wy", (o, h)), ("by", (o,))]


def n_params(n_h, n_in=N_IN, n_out=N_OUT):
    return sum(int(np.prod(s)) for _n, s in shapes(n_h, n_in, n_out))


class BatchPolicy:
    """P bo trong so, moi bo lai G con xe."""

    def __init__(self, n_hidden, device, n_in=N_IN, n_out=N_OUT):
        self.n_h = int(n_hidden)
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.device = device
        self._shapes = shapes(self.n_h, self.n_in, self.n_out)
        self.n_params = n_params(self.n_h, self.n_in, self.n_out)
        # Chuan hoa dau vao: DUNG CHUNG ca quan the va dong bang trong mot
        # the he. Moi ca the tu chuan hoa theo rieng no thi diem cua chung
        # khong con so sanh duoc voi nhau.
        self.mean = torch.zeros(self.n_in, device=device)
        self.std = torch.ones(self.n_in, device=device)

    # ------------------------------------------------------------------
    def set_norm(self, mean, var):
        self.mean = torch.as_tensor(np.asarray(mean),
                                    dtype=torch.float32).to(self.device)
        self.std = torch.as_tensor(np.sqrt(np.asarray(var) + 1e-8),
                                   dtype=torch.float32).to(self.device)

    def unpack(self, theta):
        """theta: (P, n_params) -> dict cac tensor (P, ...)."""
        out = {}
        k = 0
        for name, shape in self._shapes:
            n = int(np.prod(shape))
            out[name] = theta[:, k:k + n].reshape(theta.shape[0], *shape)
            k += n
        return out

    def new_state(self, n_pop, n_each):
        return torch.zeros(n_pop, n_each, self.n_h, device=self.device)

    def step(self, p, obs, h):
        """obs: (P,G,48) tho. h: (P,G,H). Tra ve (y (P,G,2), h moi).

        Viet bang `matmul` chu khong phai `einsum`: einsum cung ra bmm thoi,
        nhung tren card lien no hay roi ve duong vong cham. (P,G,I) x (P,I,H)
        la mot phep bmm - phep ma may nao cung lam duoc va lam nhanh.
        """
        x = (obs - self.mean) / self.std
        Tz, Tr, Tn = p["Wz"].mT, p["Wr"].mT, p["Wn"].mT
        xz, xr, xn = x @ Tz, x @ Tr, x @ Tn
        z = torch.sigmoid(xz + h @ p["Uz"].mT + p["bz"][:, None, :])
        r = torch.sigmoid(xr + h @ p["Ur"].mT + p["br"][:, None, :])
        n = torch.tanh(xn + (r * h) @ p["Un"].mT + p["bn"][:, None, :])
        h2 = (1.0 - z) * h + z * n
        y = torch.tanh(h2 @ p["Wy"].mT + p["by"][:, None, :])
        return y, h2


def to_v1(theta, n_hidden, mean, var, path, meta=None):
    """Ghi ra DUNG khuon .npz cua ban v1, de cam thang vao robot that."""
    pol = GRUPolicy(n_hidden=int(n_hidden))
    pol.set_theta(np.asarray(theta, dtype=np.float64))
    pol.norm.load(mean, var, 1e6)
    pol.save(path, meta=meta)
    return pol
