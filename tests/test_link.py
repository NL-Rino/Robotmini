"""Kiem thu duong day xe <-> wifi <-> bo nao.

Cai quan trong nhat o day: DUT KET NOI BO NAO LA MOT SU KIEN THAT. Tat tien
trinh bo nao di thi ben mo phong phai tu biet va dung lai, chu khong phai
mot co trong bo nho ai do bat len.
"""

import os
import socket
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import params as P
from sim.fleet import FleetSim
from link import protocol as proto
from link.sim_link import UdpBrainLink


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FakeBrainServer(threading.Thread):
    """Bo nao gia: tra ga co dinh, va co the tat giua chung."""

    daemon = True

    def __init__(self, port, left=0.5, right=0.4):
        super().__init__()
        self.port = port
        self.left = left
        self.right = right
        self.stop_at = None
        self.n = 0
        self.codes = {}
        self._run = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", port))
        self.sock.settimeout(0.2)

    def run(self):
        while self._run:
            try:
                data, addr = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            msg = proto.unpack(data)
            if msg is None:
                continue
            kind, rid, seq, t, payload = msg
            if kind == proto.T_HELLO:
                self.codes[rid] = payload[0]
            elif kind == proto.T_OBS:
                if self.stop_at is not None and self.n >= self.stop_at:
                    continue                       # im lang = mat nao
                self.n += 1
                self.sock.sendto(
                    proto.pack_cmd(rid, seq, t, self.left, self.right), addr)
        self.sock.close()

    def shutdown(self):
        self._run = False
        self.join(timeout=2.0)


class TestProtocol(unittest.TestCase):
    def test_goi_di_goi_ve_khong_meo(self):
        obs = [i / 100.0 for i in range(proto.N_INPUTS)]
        kind, rid, seq, t, payload = proto.unpack(proto.pack_obs(3, 77, 1.25, obs))
        self.assertEqual((kind, rid, seq), (proto.T_OBS, 3, 77))
        self.assertAlmostEqual(t, 1.25, places=5)
        for a, b in zip(obs, payload):
            self.assertAlmostEqual(a, b, places=5)

        kind, rid, _s, _t, payload = proto.unpack(proto.pack_cmd(2, 5, 0.0, -0.3, 0.8))
        self.assertEqual((kind, rid), (proto.T_CMD, 2))
        self.assertAlmostEqual(payload[0], -0.3, places=5)
        self.assertAlmostEqual(payload[1], 0.8, places=5)

        kind, rid, _s, _t, payload = proto.unpack(proto.pack_hello(1, 104))
        self.assertEqual((kind, rid, payload[0]), (proto.T_HELLO, 1, 104))

    def test_so_dau_vao_hai_ben_phai_khop(self):
        self.assertEqual(proto.N_INPUTS, P.N_INPUTS)

    def test_goi_hong_thi_bo_chu_khong_no(self):
        self.assertIsNone(proto.unpack(b""))
        self.assertIsNone(proto.unpack(b"XX" + b"\x00" * 30))
        self.assertIsNone(proto.unpack(proto.pack_obs(0, 0, 0.0, [0.0] * 48)[:20]))

    def test_goi_co_do_dai_co_dinh(self):
        self.assertEqual(len(proto.pack_obs(0, 0, 0.0, [0.0] * 48)), proto.OBS_SIZE)
        self.assertEqual(len(proto.pack_cmd(0, 0, 0.0, 0.0, 0.0)), proto.CMD_SIZE)


class TestUdpLink(unittest.TestCase):
    def test_nao_o_tien_trinh_khac_lai_duoc_xe(self):
        port = free_port()
        srv = FakeBrainServer(port, 0.6, 0.6)
        srv.start()
        try:
            sim = FleetSim(seed=3, n_robots=2)
            link = UdpBrainLink(sim.robots, "127.0.0.1", port, timeout=0.5)
            rep = sim.run(link, max_seconds=4.0)
            link.close()
            self.assertGreater(srv.n, 100, "nao phai nhan duoc cam bien")
            self.assertEqual(srv.codes, {0: sim.robots[0].code, 1: sim.robots[1].code},
                             "xe phai bao ma hoc sac cua no luc chao hoi")
            for r in sim.robots:
                self.assertGreater(r.distance, 0.5, "xe phai chay that")
            self.assertGreater(rep.sim_seconds, 3.5)
        finally:
            srv.shutdown()

    def test_tat_bo_nao_di_thi_mo_phong_dung(self):
        """Day la yeu cau chinh: chay MAI toi khi DUT KET NOI BO NAO."""
        port = free_port()
        srv = FakeBrainServer(port, 0.5, 0.5)
        srv.stop_at = 200          # tra loi 200 goi roi im lang han
        srv.start()
        try:
            sim = FleetSim(seed=3, n_robots=2)
            link = UdpBrainLink(sim.robots, "127.0.0.1", port, timeout=0.4,
                                grace=1.0)
            t0 = time.time()
            # KHONG dat max_seconds: chi bo nao im lang moi dung duoc vong lap
            rep = sim.run(link, max_seconds=None)
            wall = time.time() - t0
            link.close()
            self.assertIn("ngat ket noi", rep.stop_reason)
            self.assertLess(wall, 15.0, "phai phat hien mat nao trong vai giay")
            for r in sim.robots:
                self.assertTrue(r.brain_lost)
                self.assertAlmostEqual(r.cmd_l, 0.0, places=6, msg="mat nao thi dung banh")
                self.assertAlmostEqual(r.cmd_r, 0.0, places=6)
        finally:
            srv.shutdown()

    def test_chua_bao_gio_co_nao_thi_dung_sau_thoi_gian_cho(self):
        port = free_port()          # khong ai nghe cong nay ca
        sim = FleetSim(seed=3, n_robots=1)
        link = UdpBrainLink(sim.robots, "127.0.0.1", port, timeout=0.3, grace=0.8)
        rep = sim.run(link, max_seconds=None)
        link.close()
        self.assertIn("ngat ket noi", rep.stop_reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
