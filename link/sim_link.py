"""Phia XE: day cam bien qua UDP, cho lenh ga, va tu biet luc nao mat nao.

Day la cho "ngat ket noi bo nao" tro thanh mot su kien that chu khong phai
mot co trong bo nho: tat tien trinh bo nao di, goi tin khong ve nua, qua
`timeout` giay thi xe coi nhu mat nao, dung banh, va vong lap the gioi ket
thuc khi tat ca cac xe deu mat nao.
"""

import socket
import time

from sim.fleet import BrainLink

from . import protocol as proto


class UdpBrainLink(BrainLink):
    def __init__(self, robots, host="127.0.0.1", port=9009, timeout=0.6,
                 grace=2.0):
        self.addr = (host, int(port))
        self.timeout = float(timeout)     # bao lau khong co goi thi coi la mat
        self.grace = float(grace)         # cho bay nhieu luc dau moi tinh mat
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(self.timeout)
        self.seq = 0
        self.t0 = time.time()
        self.last_ok = {r.id: None for r in robots}
        self.ever = {r.id: False for r in robots}
        for r in robots:
            self.sock.sendto(proto.pack_hello(r.id, r.code), self.addr)

    def _age(self, robot_id):
        last = self.last_ok[robot_id]
        if last is None:
            return time.time() - self.t0 - self.grace
        return time.time() - last

    def connected(self, robot_id):
        return self._age(robot_id) < self.timeout

    def any_connected(self):
        return any(self.connected(rid) for rid in self.last_ok)

    def act(self, robot_id, obs, t):
        self.seq += 1
        try:
            self.sock.sendto(proto.pack_obs(robot_id, self.seq, t, obs), self.addr)
            while True:
                data, _ = self.sock.recvfrom(4096)
                msg = proto.unpack(data)
                if msg is None:
                    continue
                kind, rid, _seq, _tt, payload = msg
                if kind == proto.T_CMD and rid == robot_id:
                    self.last_ok[robot_id] = time.time()
                    self.ever[robot_id] = True
                    return float(payload[0]), float(payload[1])
        except (socket.timeout, OSError):
            pass
        return 0.0, 0.0

    def close(self):
        for rid in self.last_ok:
            try:
                self.sock.sendto(proto.pack_bye(rid), self.addr)
            except OSError:
                pass
        self.sock.close()
