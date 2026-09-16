# -*- coding: utf-8 -*-
"""Bo nao: mang tich chap cho anh + mang thuong cho cac so + GRU cho tri nho.

Day la cho GPU thuc su lam viec. Rieng mang nay moi buoc ton hang trieu phep
nhan, gap hang tram lan mang GRU 3.000 tham so cua ban v1 - do la ly do ban
nay dat GPU con ban kia thi khong.

Tai sao van can GRU du da co camera: doc duoc bang ma roi thi phai NHO no
trong luc quay xe di lui - luc do camera nhin ra ngoai, khong con thay bang
ma nua. Mot mang khong tri nho se quen mat minh dang cam vao hoc nao.
"""

import torch
import torch.nn as nn

from . import params as P


def _init(m, gain=2.0 ** 0.5):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    return m


class Encoder(nn.Module):
    def __init__(self, n_scalars=P.N_SCALARS, img_feat=256, sca_feat=128):
        super().__init__()
        self.conv = nn.Sequential(
            _init(nn.Conv2d(3, 32, 8, stride=4, padding=2)), nn.ReLU(),
            _init(nn.Conv2d(32, 64, 4, stride=2, padding=1)), nn.ReLU(),
            _init(nn.Conv2d(64, 64, 3, stride=1, padding=1)), nn.ReLU(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, 3, P.CAM_H, P.CAM_W)).numel()
        self.img_fc = nn.Sequential(nn.Flatten(), _init(nn.Linear(n, img_feat)),
                                    nn.ReLU())
        self.sca_fc = nn.Sequential(_init(nn.Linear(n_scalars, sca_feat)),
                                    nn.ReLU(),
                                    _init(nn.Linear(sca_feat, sca_feat)),
                                    nn.ReLU())
        self.out_dim = img_feat + sca_feat

    def forward(self, img, sca):
        return torch.cat((self.img_fc(self.conv(img)), self.sca_fc(sca)), dim=-1)


class ActorCritic(nn.Module):
    def __init__(self, n_scalars=P.N_SCALARS, hidden=256, n_act=3):
        super().__init__()
        self.enc = Encoder(n_scalars)
        self.gru = nn.GRU(self.enc.out_dim, hidden, batch_first=False)
        for name, p in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p, 1.0)
            else:
                nn.init.zeros_(p)
        self.pi = _init(nn.Linear(hidden, n_act), gain=0.01)
        self.vf = _init(nn.Linear(hidden, 1), gain=1.0)
        self.log_std = nn.Parameter(torch.full((n_act,), -0.5))
        self.hidden = hidden
        self.n_act = n_act

    def initial_state(self, n, device):
        return torch.zeros(1, n, self.hidden, device=device)

    def forward_seq(self, img, sca, h0, done=None):
        """img (T,B,3,H,W), sca (T,B,S), h0 (1,B,H). Tra ve mean, value, h."""
        T_, B = img.shape[0], img.shape[1]
        z = self.enc(img.reshape(T_ * B, *img.shape[2:]),
                     sca.reshape(T_ * B, -1)).reshape(T_, B, -1)
        if done is None:
            out, h = self.gru(z, h0)
        else:
            outs = []
            h = h0
            for t in range(T_):
                # Xoa tri nho o dung nhung moi truong vua ket thuc tap.
                h = h * (1.0 - done[t].view(1, -1, 1))
                o, h = self.gru(z[t:t + 1], h)
                outs.append(o)
            out = torch.cat(outs, dim=0)
        mean = torch.tanh(self.pi(out))
        value = self.vf(out).squeeze(-1)
        return mean, value, h

    @torch.no_grad()
    def act(self, img, sca, h, done=None):
        if done is not None:
            h = h * (1.0 - done.view(1, -1, 1))
        z = self.enc(img, sca)[None]
        out, h = self.gru(z, h)
        mean = torch.tanh(self.pi(out))[0]
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        a = dist.sample()
        logp = dist.log_prob(a).sum(-1)
        v = self.vf(out).squeeze(-1)[0]
        return torch.clamp(a, -1.0, 1.0), logp, v, h

    def evaluate(self, img, sca, h0, done, actions):
        mean, value, _h = self.forward_seq(img, sca, h0, done)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        logp = dist.log_prob(actions).sum(-1)
        ent = dist.entropy().sum(-1)
        return logp, value, ent

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
