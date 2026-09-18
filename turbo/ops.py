# -*- coding: utf-8 -*-
"""Cac phep tinh viet sao cho may nao cung chay duoc.

Ly do co file nay: card roi NVIDIA chay duoc gan het moi phep cua PyTorch,
nhung card lien (Intel HD 620 qua DirectML chang han) thi khong. Thieu mot
phep la ca chuong trinh do, va thong bao loi thuong khong noi ro thieu cai
gi. Nen: gom het cac phep "de thieu" vao day, moi may THU MOT LAN xem chay
duoc gi, roi nho lai.

Chuyen bat ngo: ba trong bon cach viet "cho may yeu" lai NHANH HON cach
thong thuong ngay tren CPU, nen chung duoc dung luon cho moi may:

    cong thuc                 cach thuong   cach o day
    can(a^2+b^2)                  0,35 ms       0,32 ms
    AND luy tien theo cot       110,78 ms      12,31 ms   (cumprod, khong cummin)
    nho nhat trong cua so         3,39 ms       0,98 ms   (bang thua, khong max_pool)

(do tren mang 1.536 x 500, CPU 4 luong)

Chi con MOT phep phai do chung: gom tia vao quat (`fan_min`). Cach nhanh
`scatter_reduce` re gap 19 lan cach thay the, nen cho nay dung phep thu:
may nao co thi dung, khong co thi di duong vong.
"""

import warnings

import torch

_CAPS = {}

# DirectML khong bao loi khi thieu mot phep - no LANG LE chep tensor sang
# CPU, tinh o do, roi chep nguoc lai. Chay thi van chay, nhung moi buoc mo
# phong phai di ve CPU mot vong. Nen o day "chay duoc nhung phai nho CPU"
# duoc tinh la KHONG co, va ta di duong vong cua minh - duong do o lai tren
# card.
_FALLBACK_WORDS = ("fall back to run on the CPU", "is not currently supported")


def caps(device):
    """Thu cac phep de thieu tren may nay. Chi thu mot lan cho moi may."""
    key = str(device)
    if key not in _CAPS:
        _CAPS[key] = _probe(device)
    return _CAPS[key]


def _try(fn):
    """Chay thu. Tra ve False neu loi, VA neu no am tham chay nho CPU."""
    try:
        with warnings.catch_warnings(record=True) as got:
            warnings.simplefilter("always")
            fn()
        for w in got:
            msg = str(w.message)
            if any(k in msg for k in _FALLBACK_WORDS):
                return False
        return True
    except Exception:
        return False


def _probe(device):
    d = torch.device(device)
    x = torch.ones(2, 4, device=d)
    i = torch.zeros(2, 4, dtype=torch.long, device=d)
    out = {}
    out["scatter_reduce"] = _try(
        lambda: torch.zeros(2, 3, device=d).scatter_reduce(
            1, i % 3, x, reduce="amin", include_self=True))
    out["float64"] = _try(lambda: x.double() + 1.0)
    out["generator"] = _try(lambda: torch.randn(
        2, device=d, generator=torch.Generator(device=d)))
    one = torch.zeros(1, dtype=torch.long, device=d)
    out["index_copy"] = _try(
        lambda: torch.zeros(2, 4, device=d).index_copy_(
            0, one, torch.ones(1, 4, device=d)))
    out["index_copy_bool"] = _try(
        lambda: torch.zeros(2, dtype=torch.bool, device=d).index_copy_(
            0, one, torch.ones(1, dtype=torch.bool, device=d)))
    out["index_fill_bool"] = _try(
        lambda: torch.zeros(2, dtype=torch.bool, device=d).index_fill_(
            0, one, False))
    out["cumprod"] = _try(lambda: x.cumprod(dim=1))
    out["bmm"] = _try(lambda: torch.bmm(torch.ones(2, 3, 4, device=d),
                                        torch.ones(2, 4, 3, device=d)))
    out["atan2"] = _try(lambda: torch.atan2(x, x))
    out["gather"] = _try(lambda: x.gather(1, i))
    out["sort"] = _try(lambda: x.sort(dim=1))
    return out


# ------------------------------------------------------------------ hinh hoc
def hypot(a, b):
    """can(a^2 + b^2). Khong dung `torch.hypot`: no de thieu tren card lien,
    ma o day khong co so nao du lon de tran, nen khong can no can than."""
    return torch.sqrt(a * a + b * b)


# ------------------------------------------------------------ chuoi theo cot
def cum_and(x, dim=1):
    """AND luy tien doc mot truc, tra ve so 0/1 (float).

    Vi 0,0 va 1,0 nhan nhau van dung tuyet doi, `cumprod` cho ket qua y het
    `cummin` - va nhanh gap chin lan.
    """
    return x.to(torch.float32).cumprod(dim=dim)


def min_window_circ(x, half):
    """Nho nhat trong cua so +-half doc truc 1, vong tron.

    Dung BANG THUA: sau j lan ghep doi, m[i] = min cua 2^j o ke tu i. Mot
    cua so rong w lop bang hai o ke nhau kich thuoc 2^j - het log2(w) buoc
    thay vi w buoc, va khong can `max_pool1d`.
    """
    n = x.shape[1]
    w = 2 * half + 1
    m = torch.cat((x[:, -half:], x, x[:, :half]), dim=1)
    L = 1
    while L * 2 <= w:
        m = torch.minimum(m[:, :-L], m[:, L:])
        L *= 2
    return torch.minimum(m[:, :n], m[:, w - L:w - L + n])


# --------------------------------------------------------------- gom vao o
FAN_MODE = "auto"          # auto | scatter | loop | cpu


def set_fan_mode(mode):
    """Chon cach gom tia vao quat. Chi de THU khi may la, dung mac dinh la
    `auto`: co `scatter_reduce` thi dung, khong thi di duong vong."""
    global FAN_MODE
    assert mode in ("auto", "scatter", "loop", "cpu"), mode
    FAN_MODE = mode


def fan_min(idx, val, n_bins, fill, device):
    """Nho nhat cua `val` trong tung o `idx`. (R,N) -> (R,n_bins)."""
    mode = FAN_MODE
    if mode == "auto":
        # Do tren HD 620 (quan the 64): cpu 930, scatter 883, loop 754
        # buoc-xe/giay. Nghia la khi card KHONG co `scatter_reduce` that
        # thi chuyen han sang CPU van hon di duong vong tren card - vi cho
        # nghen la SO LAN GOI PHEP, ma duong vong goi 48 phep con chuyen
        # sang CPU chi goi 4.
        mode = "scatter" if caps(device)["scatter_reduce"] else (
            "cpu" if torch.device(device).type != "cpu" else "loop")

    if mode == "scatter":
        out = torch.full((val.shape[0], n_bins), fill,
                         dtype=val.dtype, device=device)
        return out.scatter_reduce(1, idx, val, reduce="amin",
                                  include_self=True)

    if mode == "cpu":
        # Gom tren CPU roi tra ve. Nghe thi do, nhung card LIEN dung chung
        # thanh RAM voi CPU nen chuyen qua lai khong dat nhu card roi, va
        # duong nay chi cap phat 2 mang thay vi 25.
        i, v = idx.cpu(), val.cpu()
        out = torch.full((v.shape[0], n_bins), fill, dtype=v.dtype)
        return out.scatter_reduce(1, i, v, reduce="amin",
                                  include_self=True).to(device)

    big = torch.full_like(val, fill)
    cols = [torch.where(idx == f, val, big).min(dim=1).values
            for f in range(n_bins)]
    return torch.stack(cols, dim=1)


def dtype_audit(obj):
    """Tensor nao trong `obj` co kieu la. Tra ve danh sach (ten, kieu).

    Kieu dung la: so thuc 32 bit, dung/sai, so nguyen. Bat cu cai gi 64 BIT
    deu la lo: card lien khong lam duoc nhieu phep tren 64 bit, va no bao
    loi bang mot cau "unknown error" khong chi cho nao.
    """
    ok = (torch.float32, torch.bool, torch.int64, torch.int32, torch.uint8)
    out = []
    for k, v in sorted(vars(obj).items()):
        if torch.is_tensor(v) and v.dtype not in ok:
            out.append((k, str(v.dtype).replace("torch.", "")))
    return out


def put_rows(dst, idx, src):
    """`dst[idx] = src` theo hang, tra ve tensor ket qua.

    Tren kieu BOOL thi di vong qua so thuc: `index_fill_` tren bool thi card
    lien lam duoc, nhung `index_copy_` tren bool thi khong, va no tu choi
    bang mot cau "unknown error" khong chi cho nao ca. Cho nay chi chay mot
    lan moi the he nen doi qua doi lai khong ton gi.
    """
    if dst.dtype == torch.bool:
        f = dst.to(torch.float32)
        f.index_copy_(0, idx, src.to(torch.float32))
        return f > 0.5
    dst.index_copy_(0, idx, src)
    return dst


def any_and_first(mask, dim=1):
    """(co hay khong, chi so dau tien dung) - khong dung `max` tren kieu bool.

    `bool_tensor.max(dim)` chay tren CPU va card NVIDIA, nhung card lien thi
    hen. Doi sang so thuc truoc: ket qua y het (ca hai deu tra ve chi so DAU
    TIEN khi hoa), ma phep `max` tren so thuc thi may nao cung co.
    """
    v, i = mask.to(torch.float32).max(dim=dim)
    return v > 0.5, i


def one_hot_at(mask, col):
    """Bat DUNG mot o moi hang: o `col`, va chi khi hang do co gi trong mask.

    Thay cho `scatter_` tren kieu bool - phep do de thieu.
    """
    n = mask.shape[1]
    ar = torch.arange(n, device=mask.device)[None, :]
    return mask & (ar == col[:, None])


# ------------------------------------------------------------------ so ngau nhien
class HostRng:
    """Sinh so ngau nhien tren CPU roi chuyen sang may tinh.

    Hai cai loi, ngoai viec card lien khong co bo sinh so rieng:
      - cung mot hat giong thi MOI MAY cho ra cung mot chuoi, nen mot phien
        chay tren card so duoc voi mot phien chay tren CPU;
      - so luong sinh moi buoc rat nho (vai nghin so), nen chuyen qua lai
        khong dang ke.
    """

    def __init__(self, seed, device):
        self.device = device
        self.gen = torch.Generator()
        self.gen.manual_seed(int(seed))

    def randn(self, *shape):
        return torch.randn(*shape, generator=self.gen).to(self.device)

    def rand(self, *shape):
        return torch.rand(*shape, generator=self.gen).to(self.device)
