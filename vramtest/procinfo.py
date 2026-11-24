"""Find a service's processes via /proc: port -> PID of the server -> its GPU process tree (Linux)."""
import os


def _listening_inodes(port=None):
    """{inode: port} of listening TCP sockets."""
    result = {}
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path) as f:
                lines = f.read().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10 or parts[3] != "0A":  # 0A = LISTEN
                continue
            lport = int(parts[1].rsplit(":", 1)[1], 16)
            if port is None or lport == port:
                result[parts[9]] = lport
    return result


def _socket_inodes(pid):
    inodes = set()
    try:
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                link = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue
            if link.startswith("socket:["):
                inodes.add(link[8:-1])
    except OSError:
        pass
    return inodes


def _all_pids():
    return [int(p) for p in os.listdir("/proc") if p.isdigit()]


def pids_listening_on(port):
    inodes = _listening_inodes(port)
    if not inodes:
        return set()
    return {pid for pid in _all_pids() if _socket_inodes(pid) & inodes.keys()}


def listening_ports(pid):
    inodes = _listening_inodes()
    return sorted({inodes[i] for i in _socket_inodes(pid) if i in inodes})


def parent_map():
    parents = {}
    for pid in _all_pids():
        try:
            with open(f"/proc/{pid}/stat") as f:
                stat = f.read()
            # the process name in parentheses may contain spaces
            parents[pid] = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
    return parents


def descendants(pids):
    parents = parent_map()
    children = {}
    for child, parent in parents.items():
        children.setdefault(parent, []).append(child)
    seen, stack = set(pids), list(pids)
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def ancestors(pid):
    parents = parent_map()
    chain = []
    while pid in parents and parents[pid] > 1:
        pid = parents[pid]
        chain.append(pid)
    return chain


def cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return ""


def alive(pid):
    return os.path.exists(f"/proc/{pid}")


def resolve_service_pids(svc_cfg, gpu_pids):
    """GPU process PIDs belonging to a service -> (pids, note explaining how they were found).

    Order: explicit ``pids`` -> ``cmdline_match`` -> the process listening on ``port`` (default: the port of
    ``url`` / ``address``) and its descendants (+ ancestors, when the listener is itself a child of the engine).
    """
    gpu_pids = set(gpu_pids)
    if svc_cfg.get("pids"):
        return set(svc_cfg["pids"]), "config.pids"
    match = svc_cfg.get("cmdline_match")
    if match:
        found = {p for p in gpu_pids if match in cmdline(p)}
        return found, f"cmdline_match={match!r}"
    port = svc_cfg.get("port")
    if port is None:
        from .config import port_from
        port = port_from(svc_cfg)
    if not port:
        return set(), "no port (set url/address, port, pids or cmdline_match)"
    listeners = pids_listening_on(port)
    if not listeners:
        return set(), f"nothing listens on port {port}"
    family = descendants(listeners)
    for pid in listeners:
        family.update(ancestors(pid))
    found = family & gpu_pids
    return found, f"port {port} -> pid {sorted(listeners)}"
