"""Bo nao chay tren laptop. Nhan cam bien qua UDP, tra ga.

    python -m link.brain_server                 # bo luat viet tay
    python -m link.brain_server --port 9009

Tat tien trinh nay di la ben mo phong mat nao va dung lai - dung nhu yeu
cau "chay mai toi khi ngat ket noi bo nao".
"""

import argparse
import socket
import sys
import time

from brain.rule_brain import RuleBrain

from . import protocol as proto


def serve(host="0.0.0.0", port=9009, quiet=False, brain_factory=None):
    if brain_factory is None:
        brain_factory = lambda rid: RuleBrain(rid, seed=1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, int(port)))
    sock.settimeout(1.0)
    brains = {}
    codes = {}
    n = 0
    t_report = time.time()
    if not quiet:
        print(f"bo nao dang nghe {host}:{port} - Ctrl-C de NGAT KET NOI")
    try:
        while True:
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            msg = proto.unpack(data)
            if msg is None:
                continue
            kind, rid, seq, t, payload = msg
            if kind == proto.T_HELLO:
                brains[rid] = brain_factory(rid)
                codes[rid] = payload[0]
                if not quiet:
                    print(f"  xe {rid} ket noi, ma hoc sac {payload[0]}")
            elif kind == proto.T_OBS:
                if rid not in brains:
                    brains[rid] = brain_factory(rid)
                left, right = brains[rid](list(payload), t)
                sock.sendto(proto.pack_cmd(rid, seq, t, left, right), addr)
                n += 1
                if not quiet and time.time() - t_report > 5.0:
                    print(f"  ... {n} goi, {len(brains)} xe, t={t:.0f}s")
                    t_report = time.time()
            elif kind == proto.T_BYE:
                brains.pop(rid, None)
                if not quiet:
                    print(f"  xe {rid} chao tam biet")
    except KeyboardInterrupt:
        if not quiet:
            print("\nbo nao ngat. Ben mo phong se dung sau vai tram mili giay.")
    finally:
        sock.close()
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description="Bo nao chay tren laptop")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9009)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    serve(a.host, a.port, a.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
