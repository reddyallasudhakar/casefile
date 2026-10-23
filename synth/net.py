"""Prefer IPv4 when connecting.

Some networks hand out IPv6 addresses but don't route IPv6. Python tries addresses in
order and waits for a long timeout on each dead IPv6 one, so API calls hang for minutes.
Browsers and curl race IPv4 and IPv6 and never notice. Importing this module reorders
DNS results to put IPv4 first; IPv6 stays as a fallback.
"""

import socket

_getaddrinfo = socket.getaddrinfo


def _ipv4_first(*args, **kwargs):
    results = _getaddrinfo(*args, **kwargs)
    return sorted(results, key=lambda r: r[0] != socket.AF_INET)


if socket.getaddrinfo is _getaddrinfo:
    socket.getaddrinfo = _ipv4_first
