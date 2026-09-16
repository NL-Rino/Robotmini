"""Bo nao: GRU 48 -> H -> 2, numpy thuan, khong thu vien hoc may nao.

Kem theo bo CHUAN HOA DAU VAO. Day khong phai chi tiet vun vat: 48 dau vao
co phan bo lech nhau rat xa - co cai gan nhu luon bang 0 (den bao sac), co
cai luon quanh 0,9 (quat LiDAR nhin vao tuong). Khong chuan hoa thi moi
trong so hoc voi mot toc do khac han nhau va ES phai phi rat nhieu the he
chi de bu cai chenh lech do.

Chuan hoa chay theo kieu trung binh truot: moi vong danh gia gom them thong
ke cua cac dau vao gap phai, roi ca quan the dung CHUNG mot bo trung binh -
neu moi ca the tu chuan hoa theo rieng no thi diem cua chung khong so sanh
duoc voi nhau nua.
"""

import numpy as np

N_IN = 48
N_OUT = 2


def _sigmoid(x):
    return 0.5 * (np.tanh(0.5 * x) + 1.0)


class ObsNorm:
    """Trung binh va do lech chuan truot cua dau vao (thuat toan Welford)."""

    __slots__ = ("mean", "var", "count")

    def __init__(self, n=N_IN):
        self.mean = np.zeros(n, dtype=np.float64)
        self.var = np.ones(n, dtype=np.float64)
        self.count = 1e-4

    def update(self, batch_mean, batch_var, batch_count):
        if batch_count <= 0:
            return
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / tot)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * (self.count * batch_count / tot)
        self.var = m2 / tot
        self.count = tot

    def apply(self, obs):
        return (obs - self.mean) / np.sqrt(self.var + 1e-8)

    def state(self):
        return self.mean.copy(), self.var.copy(), float(self.count)

    def load(self, mean, var, count):
        self.mean = np.asarray(mean, dtype=np.float64).copy()
        self.var = np.asarray(var, dtype=np.float64).copy()
        self.count = float(count)


class GRUPolicy:
    """GRU mot lop. Tham so giu duoi dang MOT vector phang cho ES de xao."""

    def __init__(self, n_hidden=16, n_in=N_IN, n_out=N_OUT, seed=0):
        self.n_in = int(n_in)
        self.n_h = int(n_hidden)
        self.n_out = int(n_out)
        self.norm = ObsNorm(self.n_in)

        h, i, o = self.n_h, self.n_in, self.n_out
        self._shapes = [
            ("Wz", (h, i)), ("Wr", (h, i)), ("Wn", (h, i)),
            ("Uz", (h, h)), ("Ur", (h, h)), ("Un", (h, h)),
            ("bz", (h,)), ("br", (h,)), ("bn", (h,)),
            ("Wy", (o, h)), ("by", (o,)),
        ]
        self.n_params = sum(int(np.prod(s)) for _n, s in self._shapes)

        rng = np.random.default_rng(seed)
        parts = []
        for name, shape in self._shapes:
            if name.startswith("b"):
                parts.append(np.zeros(shape))
            else:
                fan_in = shape[1] if len(shape) == 2 else shape[0]
                parts.append(rng.normal(0.0, 1.0 / np.sqrt(fan_in), shape))
        self.theta = np.concatenate([p.ravel() for p in parts]).astype(np.float64)
        self._unpack()

    # ------------------------------------------------------------------
    def _unpack(self):
        out = {}
        k = 0
        for name, shape in self._shapes:
            n = int(np.prod(shape))
            out[name] = self.theta[k:k + n].reshape(shape)
            k += n
        self.p = out

    def set_theta(self, theta):
        self.theta = np.asarray(theta, dtype=np.float64).copy()
        self._unpack()

    def new_state(self):
        return np.zeros(self.n_h, dtype=np.float64)

    def step(self, obs, h):
        """Mot buoc. `obs` la 48 so tho; chuan hoa lam ngay o day."""
        x = self.norm.apply(np.asarray(obs, dtype=np.float64))
        p = self.p
        z = _sigmoid(p["Wz"] @ x + p["Uz"] @ h + p["bz"])
        r = _sigmoid(p["Wr"] @ x + p["Ur"] @ h + p["br"])
        n = np.tanh(p["Wn"] @ x + p["Un"] @ (r * h) + p["bn"])
        h2 = (1.0 - z) * h + z * n
        y = np.tanh(p["Wy"] @ h2 + p["by"])
        return y, h2

    # ------------------------------------------------------------------
    def saturation(self):
        """Ti le trong so lon bat thuong - dau hieu tanh sap bao hoa.

        Mot phien 5.900 the he cua ban cu tu huy vi dung day: weight_decay
        cong thang vao gradient bi Adam chuan hoa mat tac dung, trong so
        phinh dan, tanh bao hoa, va bo nao tra ve (+1,+1) voi MOI dau vao.
        """
        return float(np.mean(np.abs(self.theta) > 3.0))

    def save(self, path, meta=None):
        mean, var, count = self.norm.state()
        data = dict(theta=self.theta, n_hidden=self.n_h, n_in=self.n_in,
                    n_out=self.n_out, norm_mean=mean, norm_var=var,
                    norm_count=count)
        for k, v in (meta or {}).items():
            data["meta_" + k] = v
        np.savez(path, **data)

    @staticmethod
    def load(path):
        d = np.load(path, allow_pickle=True)
        pol = GRUPolicy(n_hidden=int(d["n_hidden"]), n_in=int(d["n_in"]),
                        n_out=int(d["n_out"]))
        pol.set_theta(d["theta"])
        if "norm_mean" in d:
            pol.norm.load(d["norm_mean"], d["norm_var"], float(d["norm_count"]))
        meta = {k[5:]: d[k] for k in d.files if k.startswith("meta_")}
        return pol, meta


class PolicyBrain:
    """Boc GRUPolicy lai cho vua khuon BrainLink cua mo phong."""

    def __init__(self, policy, robot_id=0):
        self.policy = policy
        self.id = robot_id
        self.h = policy.new_state()

    def reset(self):
        self.h = self.policy.new_state()

    def __call__(self, obs, t):
        y, self.h = self.policy.step(obs, self.h)
        return float(y[0]), float(y[1])


def factory(policy):
    return lambda rid: PolicyBrain(policy, rid)
