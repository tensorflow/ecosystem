import random

from six.moves import urllib_parse


def collect_hosts(hosts, randomize=True):
    """Collect a set of hosts and an optional chroot from a string."""
    host_ports, chroot = hosts.partition("/")[::2]
    chroot = "/" + chroot if chroot else None

    result = []
    for host_port in host_ports.split(","):
        # put all complexity of dealing with
        # IPv4 & IPv6 address:port on the urlsplit
        res = urllib_parse.urlsplit("xxx://" + host_port)
        host = res.hostname
        if host is None:
            raise ValueError("bad hostname")
        port = int(res.port) if res.port else 2181
        result.append((host.strip(), port))

    if randomize:
        random.shuffle(result)

    return result, chroot
