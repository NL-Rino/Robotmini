"""Goi tin UDP giua xe va bo nao chay tren laptop.

Gon va co dinh do dai de ban C tren ESP32 doc duoc bang mot cai struct.
Moi goi deu tu mo ta: mat mot goi khong lam hong goi sau.

  RM | ver | loai | ma_xe | _ | seq (u32) | t (f32) | tai
  HELLO  tai = ma hoc sac cua xe (u32)
  OBS    tai = 48 so f32
  CMD    tai = ga trai, ga phai (2 x f32)
  BYE    tai rong
"""

import struct

MAGIC = b"RM"
VERSION = 1

T_HELLO = 1
T_OBS = 2
T_CMD = 3
T_BYE = 4

HEADER = "<2sBBBBIf"
HEADER_SIZE = struct.calcsize(HEADER)   # 16

N_INPUTS = 48
OBS_FMT = "<%df" % N_INPUTS
CMD_FMT = "<2f"
HELLO_FMT = "<I"

OBS_SIZE = HEADER_SIZE + struct.calcsize(OBS_FMT)
CMD_SIZE = HEADER_SIZE + struct.calcsize(CMD_FMT)


def _head(kind, robot_id, seq, t):
    return struct.pack(HEADER, MAGIC, VERSION, kind, robot_id & 0xFF, 0,
                       seq & 0xFFFFFFFF, float(t))


def pack_hello(robot_id, dock_code, seq=0, t=0.0):
    return _head(T_HELLO, robot_id, seq, t) + struct.pack(HELLO_FMT,
                                                          int(dock_code) & 0xFFFFFFFF)


def pack_obs(robot_id, seq, t, obs):
    if len(obs) != N_INPUTS:
        raise ValueError(f"can dung {N_INPUTS} dau vao, nhan duoc {len(obs)}")
    return _head(T_OBS, robot_id, seq, t) + struct.pack(OBS_FMT, *[float(x) for x in obs])


def pack_cmd(robot_id, seq, t, left, right):
    return _head(T_CMD, robot_id, seq, t) + struct.pack(CMD_FMT, float(left), float(right))


def pack_bye(robot_id, seq=0, t=0.0):
    return _head(T_BYE, robot_id, seq, t)


def unpack(buf):
    """Tra ve (loai, ma_xe, seq, t, tai) hoac None neu goi hong."""
    if len(buf) < HEADER_SIZE:
        return None
    magic, ver, kind, rid, _pad, seq, t = struct.unpack(HEADER, buf[:HEADER_SIZE])
    if magic != MAGIC or ver != VERSION:
        return None
    body = buf[HEADER_SIZE:]
    if kind == T_OBS:
        if len(body) < struct.calcsize(OBS_FMT):
            return None
        payload = struct.unpack(OBS_FMT, body[:struct.calcsize(OBS_FMT)])
    elif kind == T_CMD:
        if len(body) < struct.calcsize(CMD_FMT):
            return None
        payload = struct.unpack(CMD_FMT, body[:struct.calcsize(CMD_FMT)])
    elif kind == T_HELLO:
        if len(body) < struct.calcsize(HELLO_FMT):
            return None
        payload = struct.unpack(HELLO_FMT, body[:struct.calcsize(HELLO_FMT)])
    else:
        payload = ()
    return kind, rid, seq, t, payload
