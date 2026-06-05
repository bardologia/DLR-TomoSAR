from __future__ import annotations

import os
import pwd
import shutil
import socket
import subprocess
import threading
import time


class SystemMonitor:

    GPU_QUERY    = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,uuid"
    PROC_LIMIT   = 30
    DU_REFRESH_S = 600.0

    def __init__(self, paths) -> None:
        self.paths       = paths
        self.lock        = threading.Lock()
        self.prev_cpu    = {}
        self.prev_proc   = {}
        self.prev_proc_t = 0.0
        self.uid         = os.getuid()
        self.user        = pwd.getpwuid(self.uid).pw_name
        self.clk         = os.sysconf("SC_CLK_TCK")
        self.page        = os.sysconf("SC_PAGE_SIZE")
        self.user_root   = self._user_root()
        self.du_usage    = {"user": None, "repo": None}

        threading.Thread(target=self._du_loop, daemon=True).start()

    def snapshot(self) -> dict:
        gpu_mem, gpu_users = self._gpu_procs()

        with self.lock:
            cores, total = self._cpu_percents()
            procs        = self._procs(gpu_mem)

        return {
            "host"   : socket.gethostname(),
            "user"   : self.user,
            "uptime" : self._uptime(),
            "cpu"    : {"count": os.cpu_count() or len(cores), "total": total, "cores": cores, "load": list(os.getloadavg())},
            "mem"    : self._memory(),
            "disk"   : self._disk(),
            "gpus"   : self._gpus(gpu_users),
            "procs"  : procs,
        }

    def _cpu_percents(self) -> tuple[list[float], float]:
        cores = []
        total = 0.0

        try:
            lines = open("/proc/stat").read().splitlines()
        except OSError:
            return cores, total

        for line in lines:
            if not line.startswith("cpu"):
                continue
            parts = line.split()
            key   = parts[0]
            vals  = [int(v) for v in parts[1:9]]
            busy  = sum(vals) - vals[3] - vals[4]
            whole = sum(vals)

            prev = self.prev_cpu.get(key)
            self.prev_cpu[key] = (busy, whole)

            pct = 0.0
            if prev is not None and whole > prev[1]:
                pct = round(100.0 * (busy - prev[0]) / (whole - prev[1]), 1)

            if key == "cpu":
                total = pct
            else:
                cores.append(pct)

        return cores, total

    def _procs(self, gpu_mem: dict) -> list[dict]:
        now  = time.monotonic()
        dt   = now - self.prev_proc_t
        rows = []
        seen = set()

        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                if os.stat(f"/proc/{pid}").st_uid != self.uid:
                    continue
                stat = open(f"/proc/{pid}/stat").read()
            except OSError:
                continue

            try:
                close  = stat.rindex(")")
                comm   = stat[stat.index("(") + 1 : close]
                fields = stat[close + 2 :].split()
                state  = fields[0]
                jiff   = int(fields[11]) + int(fields[12])
                rss    = int(fields[21]) * self.page
            except (ValueError, IndexError):
                continue

            prev = self.prev_proc.get(pid)
            cpu  = 0.0
            if prev is not None and self.prev_proc_t > 0 and dt > 0:
                cpu = max(0.0, round(100.0 * (jiff - prev) / self.clk / dt, 1))
            self.prev_proc[pid] = jiff
            seen.add(pid)

            try:
                raw = open(f"/proc/{pid}/cmdline", "rb").read()
                cmd = raw.replace(b"\x00", b" ").decode(errors="replace").strip()
            except OSError:
                cmd = ""

            rows.append({
                "pid"   : pid,
                "state" : state,
                "cpu"   : cpu,
                "rss"   : rss,
                "gpu"   : gpu_mem.get(pid, 0),
                "cmd"   : (cmd or comm)[:200],
            })

        self.prev_proc_t = now
        self.prev_proc   = {p: j for p, j in self.prev_proc.items() if p in seen}

        rows.sort(key=lambda r: (-r["cpu"], -r["gpu"], -r["rss"]))
        return rows[: self.PROC_LIMIT]

    def _gpu_procs(self) -> tuple[dict, dict]:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory,gpu_uuid", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {}, {}

        if out.returncode != 0:
            return {}, {}

        usage = {}
        users = {}
        for line in out.stdout.strip().splitlines():
            cells = [c.strip() for c in line.split(",")]
            if len(cells) < 3:
                continue
            try:
                pid = int(cells[0])
                mem = float(cells[1])
            except ValueError:
                continue

            usage[pid] = usage.get(pid, 0) + mem

            entry = users.setdefault(cells[2], {"mine": False, "others": False})
            try:
                mine = os.stat(f"/proc/{pid}").st_uid == self.uid
            except OSError:
                mine = False
            entry["mine" if mine else "others"] = True

        return usage, users

    def _memory(self) -> dict:
        info = {}
        try:
            for line in open("/proc/meminfo").read().splitlines():
                key, _, rest = line.partition(":")
                info[key]    = int(rest.split()[0]) * 1024
        except (OSError, ValueError, IndexError):
            return {}

        return {
            "total"      : info.get("MemTotal", 0),
            "available"  : info.get("MemAvailable", 0),
            "swap_total" : info.get("SwapTotal", 0),
            "swap_free"  : info.get("SwapFree", 0),
        }

    def _disk(self) -> dict:
        try:
            usage = shutil.disk_usage(self.paths.repo_root)
        except OSError:
            return {}

        return {
            "path"      : str(self.paths.repo_root),
            "total"     : usage.total,
            "used"      : usage.used,
            "free"      : usage.free,
            "user_path" : str(self.user_root),
            "user_used" : self.du_usage["user"],
            "repo_path" : str(self.paths.repo_root),
            "repo_used" : self.du_usage["repo"],
        }

    def _user_root(self):
        current = self.paths.repo_root.resolve()

        while current != current.parent:
            parent = current.parent
            try:
                if os.stat(parent).st_uid != self.uid:
                    break
            except OSError:
                break
            current = parent

        return current

    def _du_loop(self) -> None:
        while True:
            self.du_usage["repo"] = self._du(self.paths.repo_root)
            self.du_usage["user"] = self._du(self.user_root) if self.user_root != self.paths.repo_root else self.du_usage["repo"]
            time.sleep(self.DU_REFRESH_S)

    def _du(self, path) -> int | None:
        try:
            out = subprocess.run(["du", "-s", "--block-size=1", str(path)], capture_output=True, text=True, timeout=900)
        except (OSError, subprocess.TimeoutExpired):
            return None

        try:
            return int(out.stdout.split()[0])
        except (ValueError, IndexError):
            return None

    def _uptime(self) -> float:
        try:
            return float(open("/proc/uptime").read().split()[0])
        except (OSError, ValueError, IndexError):
            return 0.0

    def _gpus(self, gpu_users: dict | None = None) -> list[dict]:
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={self.GPU_QUERY}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
        except (OSError, subprocess.TimeoutExpired):
            return []

        if out.returncode != 0:
            return []

        gpus = []
        for line in out.stdout.strip().splitlines():
            cells = [c.strip() for c in line.split(",")]
            if len(cells) < 9:
                continue
            occupancy = (gpu_users or {}).get(cells[8], {})
            gpus.append({
                "index"       : self._num(cells[0]),
                "name"        : cells[1],
                "util"        : self._num(cells[2]),
                "mem_used"    : self._num(cells[3]),
                "mem_total"   : self._num(cells[4]),
                "temp"        : self._num(cells[5]),
                "power"       : self._num(cells[6]),
                "power_limit" : self._num(cells[7]),
                "mine"        : bool(occupancy.get("mine")),
                "others"      : bool(occupancy.get("others")),
            })
        return gpus

    def _num(self, raw: str):
        try:
            value = float(raw)
        except ValueError:
            return None
        return int(value) if value.is_integer() else value
