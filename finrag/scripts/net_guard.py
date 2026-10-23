"""
Outbound-network guard: inside `outbound_only(hosts)` the process may resolve and
connect only to the listed hosts (plus loopback, for the local model server).
Anything else raises NetworkBlocked and is recorded in `guard.blocked`.

portfolio_sync runs every sync inside it with the adapter's `allowed_hosts`, so a
broker adapter can reach its broker and nothing else. Tests use it to prove that
query-time answering makes no outbound connection at all.

It patches socket.getaddrinfo / socket.socket.connect for the whole process while
active, so use it in CLI runs and tests, not around a multi-threaded server.
"""
from __future__ import annotations

import contextlib
import ipaddress
import socket
from dataclasses import dataclass, field


class NetworkBlocked(RuntimeError):
    pass


def _is_loopback(host: str) -> bool:
    if host in ("localhost", "localhost.localdomain"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass
class Guard:
    allowed: set[str]
    allow_loopback: bool
    blocked: list[str] = field(default_factory=list)
    contacted: list[str] = field(default_factory=list)
    _ips: set[str] = field(default_factory=set)


@contextlib.contextmanager
def outbound_only(hosts=(), allow_loopback: bool = True):
    guard = Guard({h.lower() for h in hosts}, allow_loopback)
    real_getaddrinfo, real_connect = socket.getaddrinfo, socket.socket.connect

    def getaddrinfo(host, *args, **kwargs):
        name = (host.decode() if isinstance(host, bytes) else str(host or "")).lower()
        if guard.allow_loopback and _is_loopback(name):
            return real_getaddrinfo(host, *args, **kwargs)
        if name not in guard.allowed:
            guard.blocked.append(name)
            raise NetworkBlocked(f"outbound connection to {name} blocked (allowed: {', '.join(sorted(guard.allowed)) or 'none'})")
        guard.contacted.append(name)
        result = real_getaddrinfo(host, *args, **kwargs)
        guard._ips.update(str(r[4][0]) for r in result)
        return result

    def connect(self, address):
        if isinstance(address, tuple):
            ip = str(address[0])
            if not ((guard.allow_loopback and _is_loopback(ip)) or ip in guard._ips):
                guard.blocked.append(ip)
                raise NetworkBlocked(f"outbound connection to {ip} blocked")
        return real_connect(self, address)  # AF_UNIX paths are local

    socket.getaddrinfo, socket.socket.connect = getaddrinfo, connect
    try:
        yield guard
    finally:
        socket.getaddrinfo, socket.socket.connect = real_getaddrinfo, real_connect
