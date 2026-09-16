"""Evolution Strategies - ban da sua bon cho quan trong nhat.

Ban cu chay 5.900 the he voi quan the 48 va 3 tap moi ca the, roi van thua
bo luat viet tay. Khong phai vi ES yeu, ma vi BON CHO sau day:

1. CHUNG SO NGAU NHIEN (common random numbers).
   Ban cu cham diem moi ca the bang 3 tap RIENG cua no. Chenh lech diem giua
   hai ca the khi do phan lon la chenh lech VAN MAY (ai gap mat bang de hon,
   ai gap nguoi di lai chan duong), chu khong phai chenh lech nang luc. ES
   xep hang theo diem, nen no xep hang theo van may -> gradient la nhieu.
   Sua: trong MOT the he, MOI ca the chay tren DUNG mot bo hat giong - cung
   mat bang, cung cho dat xe, cung nhieu cam bien. Khi do hieu diem chi con
   phan do trong so gay ra. Sang the he sau thi doi hat giong.
   Day la thu re nhat va an thua nhat trong ca danh sach.

2. XAO DOI GUONG (mirrored sampling).
   Voi moi vector nhieu eps, danh gia CA HAI phia: theta+sigma*eps va
   theta-sigma*eps. Uoc luong gradient thanh (f+ - f-)/2 * eps, tu triet
   tieu phan chung va khong can moc so sanh. Phuong sai giam khoang mot
   nua ma khong ton them thong tin nao.

3. CHUAN HOA DAU VAO.
   48 dau vao co phan bo lech nhau rat xa. Chuan hoa bang trung binh truot
   dung CHUNG cho ca quan the (dong bang trong mot the he, cap nhat o cuoi
   the he). Day la thu lam cho tim kiem ngau nhien don gian duoi kip cac
   phuong phap hoc tang cuong phuc tap.

4. ADAMW, KHONG PHAI ADAM + WEIGHT_DECAY.
   Loi (l) cua ban cu: weight_decay cong thang vao gradient thi bi Adam
   chuan hoa luon, het tac dung keo trong so ve 0; trong so phinh dan, tanh
   bao hoa, bo nao tra ve (+1,+1) voi moi dau vao. Mot phien 5.900 the he
   tu huy vi day. AdamW tru phan suy giam RIENG, sau khi da chia cho
   sqrt(v), nen no lam dung viec cua no.

Cong them: xep hang giua (centered rank) co CHAN - ca quan the diem bang
nhau thi tra ve vector 0, khong sinh gradient tu hu khong.
"""

import numpy as np


def centered_ranks(x):
    """Doi diem thanh hang trong khoang [-0,5; 0,5].

    Nho vay thang do phan thuong khong con anh huong toi buoc cap nhat, va
    mot ca the an may cung khong keo ca quan the di theo.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 2 or float(x.max() - x.min()) < 1e-12:
        # CHAN: ca quan the diem y het nhau thi KHONG co thong tin gi ca.
        # Ban cu o day van tra ve mot vector hang va di mot buoc vao cho
        # khong dau, moi lan mot it, cho toi khi trong so phinh to.
        return np.zeros(n, dtype=np.float64)
    ranks = np.empty(n, dtype=np.float64)
    ranks[np.argsort(x)] = np.arange(n, dtype=np.float64)
    return ranks / (n - 1.0) - 0.5


class AdamW:
    """Adam voi phan suy giam trong so TACH RIENG."""

    def __init__(self, n, lr=0.02, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0.005):
        self.lr = float(lr)
        self.b1 = float(beta1)
        self.b2 = float(beta2)
        self.eps = float(eps)
        self.wd = float(weight_decay)
        self.m = np.zeros(n, dtype=np.float64)
        self.v = np.zeros(n, dtype=np.float64)
        self.t = 0

    def step(self, theta, grad):
        """`grad` la huong LAM TANG diem; ham nay di theo huong do."""
        self.t += 1
        self.m = self.b1 * self.m + (1.0 - self.b1) * grad
        self.v = self.b2 * self.v + (1.0 - self.b2) * grad * grad
        mh = self.m / (1.0 - self.b1 ** self.t)
        vh = self.v / (1.0 - self.b2 ** self.t)
        # AdamW: phan suy giam nam NGOAI phep chia cho sqrt(v).
        return theta + self.lr * mh / (np.sqrt(vh) + self.eps) - self.lr * self.wd * theta

    def state(self):
        return dict(m=self.m, v=self.v, t=self.t, lr=self.lr, wd=self.wd)

    def load(self, d):
        self.m = np.asarray(d["m"], dtype=np.float64).copy()
        self.v = np.asarray(d["v"], dtype=np.float64).copy()
        self.t = int(d["t"])
        self.lr = float(d["lr"])
        self.wd = float(d["wd"])


def noise(n_params, gen, index):
    """Sinh lai dung vector nhieu tu hat giong.

    Nho vay tien trinh con chi can nhan mot con so nguyen thay vi ca vector
    3.000 so, va tien trinh cha dung lai duoc dung vector do de gop gradient.
    """
    rng = np.random.default_rng((gen * 1_000_003 + index) & 0x7FFFFFFF)
    return rng.standard_normal(n_params)


def gradient(n_params, gen, scores_plus, scores_minus, sigma):
    """Uoc luong huong lam tang diem tu cac cap doi guong."""
    n_pairs = len(scores_plus)
    both = np.concatenate([np.asarray(scores_plus, dtype=np.float64),
                           np.asarray(scores_minus, dtype=np.float64)])
    r = centered_ranks(both)
    rp = r[:n_pairs]
    rm = r[n_pairs:]
    g = np.zeros(n_params, dtype=np.float64)
    w = rp - rm
    if not np.any(w):
        return g
    for i in range(n_pairs):
        if w[i] != 0.0:
            g += w[i] * noise(n_params, gen, i)
    return g / (n_pairs * sigma)
