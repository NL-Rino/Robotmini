# -*- coding: utf-8 -*-
"""PPO co hoi quy, chay hoan toan tren GPU.

Vi sao doi tu ES (ban v1) sang PPO o ban nay: ES khong dung duong lan truyen
nguoc, no chi cham diem roi xao trong so. Voi mang 3.000 tham so thi cach do
re va song song hoa theo nhan CPU rat tot. Voi mang 1,4 trieu tham so va dau
vao la anh thi nguoc lai: khong co gradient that thi so lan thu can thiet
tang theo so chieu, ma 1,4 trieu chieu thi khong kham noi.

Anh duoc GIU DUOI DANG uint8 trong bo dem. Doi sang float32 ngay luc thu
thap thi bo dem phinh gap bon lan - va bo dem la thu an VRAM nhieu nhat.
"""

import torch
import torch.nn as nn


class Buffer:
    def __init__(self, T, B, img_shape, n_sca, n_act, device):
        self.T, self.B = T, B
        z = lambda *s, **k: torch.zeros(*s, device=device, **k)
        self.img = z(T, B, *img_shape, dtype=torch.uint8)
        self.sca = z(T, B, n_sca)
        self.act = z(T, B, n_act)
        self.logp = z(T, B)
        self.val = z(T, B)
        self.rew = z(T, B)
        self.done = z(T, B)
        self.h0 = None

    def gae(self, last_val, gamma=0.99, lam=0.95):
        adv = torch.zeros_like(self.rew)
        run = torch.zeros_like(last_val)
        for t in reversed(range(self.T)):
            nxt = last_val if t == self.T - 1 else self.val[t + 1]
            nonterm = 1.0 - self.done[t]
            delta = self.rew[t] + gamma * nxt * nonterm - self.val[t]
            run = delta + gamma * lam * nonterm * run
            adv[t] = run
        return adv, adv + self.val


class PPO:
    def __init__(self, model, device, lr=2.5e-4, clip=0.2, epochs=4,
                 minibatches=4, ent_coef=0.004, vf_coef=0.5, max_grad=0.5,
                 gamma=0.99, lam=0.95, amp=False):
        self.model = model
        self.device = device
        self.opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
        self.clip = clip
        self.epochs = epochs
        self.minibatches = minibatches
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad = max_grad
        self.gamma = gamma
        self.lam = lam
        # Nua do chinh xac CHI ap cho mang - khong ap cho phan doi tia. Doi
        # tia ma chay o float16 thi cac phep chia trong phep thu theo lop mat
        # chinh xac va tuong bat dau ro ri.
        self.amp = bool(amp) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

    def update(self, buf, last_val):
        adv, ret = buf.gae(last_val, self.gamma, self.lam)
        B = buf.B
        n_mb = max(1, min(self.minibatches, B))
        size = B // n_mb
        stats = dict(pi=0.0, vf=0.0, ent=0.0, kl=0.0, clipped=0.0, n=0)

        for _ep in range(self.epochs):
            perm = torch.randperm(B, device=self.device)
            for i in range(n_mb):
                idx = perm[i * size:(i + 1) * size]
                if idx.numel() == 0:
                    continue
                img = buf.img[:, idx].float() / 255.0
                sca = buf.sca[:, idx]
                # Moi lo nho la mot nhom MOI TRUONG voi CA CHUOI thoi gian
                # cua no. Cat theo thoi gian thi trang thai GRU dut, va bo
                # nao se hoc mot bai toan khong phai bai toan that.
                with torch.autocast("cuda", enabled=self.amp):
                    logp, val, ent = self.model.evaluate(
                        img, sca, buf.h0[:, idx].contiguous(),
                        buf.done[:, idx], buf.act[:, idx])
                logp = logp.float()
                val = val.float()
                ent = ent.float()

                a = adv[:, idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = torch.exp(logp - buf.logp[:, idx])
                l1 = ratio * a
                l2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * a
                pi_loss = -torch.minimum(l1, l2).mean()

                r = ret[:, idx]
                v_clip = buf.val[:, idx] + torch.clamp(
                    val - buf.val[:, idx], -self.clip, self.clip)
                vf_loss = 0.5 * torch.maximum((val - r) ** 2,
                                              (v_clip - r) ** 2).mean()
                ent_loss = ent.mean()
                loss = pi_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss

                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad)
                self.scaler.step(self.opt)
                self.scaler.update()

                with torch.no_grad():
                    stats["pi"] += float(pi_loss)
                    stats["vf"] += float(vf_loss)
                    stats["ent"] += float(ent_loss)
                    stats["kl"] += float((buf.logp[:, idx] - logp).mean())
                    stats["clipped"] += float(
                        ((ratio - 1.0).abs() > self.clip).float().mean())
                    stats["n"] += 1
        n = max(1, stats.pop("n"))
        return {k: v / n for k, v in stats.items()}


@torch.no_grad()
def collect(env, model, buf, state, device, amp=False):
    """Chay T buoc, ghi vao bo dem. Khong mot lan dong bo GPU-CPU nao o day."""
    h, done = state["h"], state["done"]
    buf.h0 = h.clone()
    img, sca = env.observe()
    use_amp = amp and device.type == "cuda"
    for t in range(buf.T):
        with torch.autocast("cuda", enabled=use_amp):
            a, logp, v, h = model.act(img, sca, h, done)
        a, logp, v = a.float(), logp.float(), v.float()
        h = h.float()
        buf.img[t] = (img * 255).to(torch.uint8)
        buf.sca[t] = sca
        buf.act[t] = a
        buf.logp[t] = logp
        buf.val[t] = v
        rew, d, _info = env.step(a)
        buf.rew[t] = rew
        buf.done[t] = d.float()
        done = d.float()
        env.reset_idx(d)
        img, sca = env.observe()
    with torch.autocast("cuda", enabled=use_amp):
        _a, _lp, last_v, h = model.act(img, sca, h, done)
    state["h"], state["done"] = h.float(), done
    return last_v.float()
