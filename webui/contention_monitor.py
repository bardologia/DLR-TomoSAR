from __future__ import annotations

import os
import threading
import time
from collections import deque
from datetime    import datetime

from web_logger import WebLogger


class ContentionMonitor:

    INTERVAL     = 5.0
    SECTOR       = 512
    SHARE        = 0.5
    MEM_PSI_SOME = 10.0
    MEM_PSI_FULL = 3.0
    IO_PSI_SOME  = 25.0
    IO_UTIL      = 80.0
    CPU_PSI_SOME = 40.0
    LOAD_RATIO   = 1.0
    SWAP_OUT_MBS = 0.5
    CLEAR_MARGIN = 0.8
    SUSTAIN      = 2

    def __init__(self, paths, logger: WebLogger) -> None:
        self.paths   = paths
        self.logger  = logger
        self.lock    = threading.Lock()
        self.armed   = False
        self.uid     = os.getuid()
        self.clk     = os.sysconf("SC_CLK_TCK")
        self.page    = os.sysconf("SC_PAGE_SIZE")
        self.cores   = os.cpu_count() or 1
        self.device  = self._resolve_device()

        self.signals = {}
        self.active  = {}
        self.events  = deque(maxlen=20)
        self.streaks = {}

        self.prev_swap = None
        self.prev_disk = None
        self.prev_mine = None
        self.prev_t    = 0.0

    def start(self) -> None:
        worker = threading.Thread(target=self._watch, name="ContentionMonitor", daemon=True)
        worker.start()
        self.armed = True
        self.logger.muted(f"contention monitor armed (device {self.device or 'n/a'}, share {self.SHARE:.0%})")

    def state(self) -> dict:
        limits = {
            "mem_psi_some" : self.MEM_PSI_SOME,
            "io_psi_some"  : self.IO_PSI_SOME,
            "cpu_psi_some" : self.CPU_PSI_SOME,
            "load_ratio"   : self.LOAD_RATIO,
            "swap_out_mbs" : self.SWAP_OUT_MBS,
            "share"        : self.SHARE,
            "interval"     : self.INTERVAL,
        }

        with self.lock:
            return {
                "armed"     : self.armed,
                "device"    : self.device,
                "limits"    : limits,
                "signals"   : dict(self.signals),
                "impacting" : any(a["mine"] for a in self.active.values()),
                "active"    : list(self.active.values()),
                "events"    : list(self.events),
            }

    def _watch(self) -> None:
        while True:
            time.sleep(self.INTERVAL)
            try:
                signals = self._sample()
                with self.lock:
                    self.signals = signals
                self._evaluate(signals)
            except Exception as exc:
                self.logger.error(f"contention monitor error: {exc}")

    def _sample(self) -> dict:
        now = time.monotonic()
        dt  = max(now - self.prev_t, 1e-6) if self.prev_t else self.INTERVAL

        psi  = self._read_psi()
        swap = self._read_swap(dt)
        mem  = self._read_memory()
        disk = self._read_disk(dt)
        mine = self._read_mine(dt)

        self.prev_t = now

        mem_used_gb  = mem["used_gb"]
        mem_share    = (mine["rss_gb"] / mem_used_gb) if mem_used_gb > 0 else 0.0
        cpu_capacity = self.cores * 100.0
        cpu_share    = (mine["cpu_pct"] / cpu_capacity) if cpu_capacity > 0 else 0.0
        io_share     = (mine["io_mbs"] / disk["total_mbs"]) if disk["total_mbs"] > 0.05 else 0.0

        return {
            "psi"  : psi,
            "swap" : swap,
            "mem"  : {"used_pct": mem["used_pct"], "mine_gb": mine["rss_gb"], "mine_share": min(mem_share, 1.0)},
            "io"   : {**disk, "mine_mbs": mine["io_mbs"], "mine_share": min(io_share, 1.0)},
            "cpu"  : {"load_ratio": os.getloadavg()[0] / self.cores, "mine_pct": mine["cpu_pct"], "mine_share": min(cpu_share, 1.0)},
            "mine" : {"nproc": mine["nproc"], "rss_gb": mine["rss_gb"]},
        }

    def _read_psi(self) -> dict:
        out = {}
        for resource in ("cpu", "memory", "io"):
            some, full = 0.0, 0.0
            try:
                for line in open(f"/proc/pressure/{resource}").read().splitlines():
                    fields = dict(token.split("=") for token in line.split()[1:] if "=" in token)
                    if line.startswith("some"):
                        some = float(fields.get("avg10", 0.0))
                    elif line.startswith("full"):
                        full = float(fields.get("avg10", 0.0))
            except (OSError, ValueError):
                pass
            out["mem" if resource == "memory" else resource] = {"some": some, "full": full}
        return out

    def _read_swap(self, dt: float) -> dict:
        pin, pout = 0, 0
        try:
            for line in open("/proc/vmstat").read().splitlines():
                if line.startswith("pswpin "):
                    pin = int(line.split()[1])
                elif line.startswith("pswpout "):
                    pout = int(line.split()[1])
        except (OSError, ValueError):
            return {"in_mbs": 0.0, "out_mbs": 0.0}

        prev           = self.prev_swap
        self.prev_swap = (pin, pout)
        if prev is None:
            return {"in_mbs": 0.0, "out_mbs": 0.0}

        in_mbs  = max(0, pin - prev[0])  * self.page / dt / (1024.0 ** 2)
        out_mbs = max(0, pout - prev[1]) * self.page / dt / (1024.0 ** 2)
        return {"in_mbs": in_mbs, "out_mbs": out_mbs}

    def _read_memory(self) -> dict:
        info = {}
        try:
            for line in open("/proc/meminfo").read().splitlines():
                key, _, rest = line.partition(":")
                info[key]    = int(rest.split()[0]) * 1024
        except (OSError, ValueError, IndexError):
            return {"used_pct": 0.0, "used_gb": 0.0}

        total     = info.get("MemTotal", 0)
        available = info.get("MemAvailable", 0)
        used      = max(0, total - available)
        return {"used_pct": (100.0 * used / total) if total else 0.0, "used_gb": used / (1024.0 ** 3)}

    def _read_disk(self, dt: float) -> dict:
        empty = {"device": self.device, "util": 0.0, "await_ms": 0.0, "read_mbs": 0.0, "write_mbs": 0.0, "total_mbs": 0.0}
        if self.device is None:
            return empty

        row = self._diskstat_row()
        if row is None:
            return empty

        reads, sec_read, ms_read, writes, sec_write, ms_write, io_ticks = row
        prev           = self.prev_disk
        self.prev_disk = row
        if prev is None:
            return empty

        d_reads  = max(0, reads     - prev[0])
        d_secr   = max(0, sec_read  - prev[1])
        d_msr    = max(0, ms_read   - prev[2])
        d_writes = max(0, writes    - prev[3])
        d_secw   = max(0, sec_write - prev[4])
        d_msw    = max(0, ms_write  - prev[5])
        d_ticks  = max(0, io_ticks  - prev[6])

        ops       = d_reads + d_writes
        read_mbs  = d_secr * self.SECTOR / dt / (1024.0 ** 2)
        write_mbs = d_secw * self.SECTOR / dt / (1024.0 ** 2)
        return {
            "device"    : self.device,
            "util"      : min(100.0, d_ticks / (dt * 1000.0) * 100.0),
            "await_ms"  : ((d_msr + d_msw) / ops) if ops else 0.0,
            "read_mbs"  : read_mbs,
            "write_mbs" : write_mbs,
            "total_mbs" : read_mbs + write_mbs,
        }

    def _diskstat_row(self):
        try:
            lines = open("/proc/diskstats").read().splitlines()
        except OSError:
            return None

        for line in lines:
            parts = line.split()
            if len(parts) < 14 or parts[2] != self.device:
                continue
            nums = [int(v) for v in parts[3:]]
            return (nums[0], nums[2], nums[3], nums[4], nums[6], nums[7], nums[9])
        return None

    def _read_mine(self, dt: float) -> dict:
        rss        = 0
        jiffies    = 0
        io_bytes   = 0
        nproc      = 0

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
                fields   = stat[stat.rindex(")") + 2:].split()
                jiffies += int(fields[11]) + int(fields[12])
                rss     += int(fields[21]) * self.page
                nproc   += 1
            except (ValueError, IndexError):
                continue

            io_bytes += self._pid_io(pid)

        prev           = self.prev_mine
        self.prev_mine = (jiffies, io_bytes)

        cpu_pct = 0.0
        io_mbs  = 0.0
        if prev is not None:
            cpu_pct = max(0.0, 100.0 * (jiffies - prev[0]) / self.clk / dt)
            io_mbs  = max(0.0, (io_bytes - prev[1]) / dt / (1024.0 ** 2))

        return {"nproc": nproc, "rss_gb": rss / (1024.0 ** 3), "cpu_pct": cpu_pct, "io_mbs": io_mbs}

    def _pid_io(self, pid: int) -> int:
        try:
            total = 0
            for line in open(f"/proc/{pid}/io").read().splitlines():
                if line.startswith("read_bytes:") or line.startswith("write_bytes:"):
                    total += int(line.split()[1])
            return total
        except (OSError, ValueError, IndexError):
            return 0

    def _evaluate(self, signals: dict) -> None:
        psi  = signals["psi"]
        swap = signals["swap"]
        mem  = signals["mem"]
        disk = signals["io"]
        cpu  = signals["cpu"]

        mem_breach = psi["mem"]["some"] >= self.MEM_PSI_SOME or swap["out_mbs"] >= self.SWAP_OUT_MBS
        mem_clear  = psi["mem"]["some"] < self.MEM_PSI_SOME * self.CLEAR_MARGIN and swap["out_mbs"] < self.SWAP_OUT_MBS
        mem_mine   = mem["mine_share"] >= self.SHARE
        mem_level  = "danger" if (psi["mem"]["full"] >= self.MEM_PSI_FULL or swap["out_mbs"] >= self.SWAP_OUT_MBS) else "warn"
        swap_note  = f", swapping out {swap['out_mbs']:.1f} MB/s" if swap["out_mbs"] >= self.SWAP_OUT_MBS else ""
        self._track("mem", mem_level, mem_breach, mem_clear, mem_mine,
                    f"memory contention: tasks stalled {psi['mem']['some']:.0f}% of the time{swap_note}; "
                    f"you hold {mem['mine_share']:.0%} of used RAM ({mem['mine_gb']:.1f} GB)")

        io_breach = psi["io"]["some"] >= self.IO_PSI_SOME or disk["util"] >= self.IO_UTIL
        io_clear  = psi["io"]["some"] < self.IO_PSI_SOME * self.CLEAR_MARGIN and disk["util"] < self.IO_UTIL * self.CLEAR_MARGIN
        io_mine   = disk["mine_share"] >= self.SHARE
        io_level  = "danger" if psi["io"]["full"] >= self.MEM_PSI_FULL else "warn"
        self._track("io", io_level, io_breach, io_clear, io_mine,
                    f"disk contention on {disk['device']}: {disk['util']:.0f}% busy, tasks stalled {psi['io']['some']:.0f}%; "
                    f"you drive {disk['mine_share']:.0%} of the I/O ({disk['mine_mbs']:.0f} MB/s)")

        cpu_breach = psi["cpu"]["some"] >= self.CPU_PSI_SOME or cpu["load_ratio"] >= self.LOAD_RATIO
        cpu_clear  = psi["cpu"]["some"] < self.CPU_PSI_SOME * self.CLEAR_MARGIN and cpu["load_ratio"] < self.LOAD_RATIO * self.CLEAR_MARGIN
        cpu_mine   = cpu["mine_share"] >= self.SHARE
        self._track("cpu", "warn", cpu_breach, cpu_clear, cpu_mine,
                    f"CPU saturated: load/core {cpu['load_ratio']:.2f}, tasks stalled {psi['cpu']['some']:.0f}%; "
                    f"you use {cpu['mine_share']:.0%} of all cores ({cpu['mine_pct']:.0f}% busy)")

    def _track(self, kind: str, level: str, breached: bool, cleared: bool, mine: bool, message: str) -> None:
        if breached:
            self.streaks[kind] = self.streaks.get(kind, 0) + 1
            if self.streaks[kind] >= self.SUSTAIN:
                self._raise(kind, level, mine, message)
        else:
            self.streaks[kind] = 0
            if cleared:
                self._drop(kind)

    def _raise(self, kind: str, level: str, mine: bool, message: str) -> None:
        with self.lock:
            known = self.active.get(kind)
            if known is None:
                self.active[kind] = {"kind": kind, "level": level, "mine": mine, "message": message, "since": datetime.now().isoformat(timespec="seconds")}
                if mine:
                    self.events.append({"kind": kind, "level": level, "message": message, "time": datetime.now().isoformat(timespec="seconds")})
                    self.logger.warning(f"neighbour impact: {message}")
            else:
                known["level"]   = level
                known["mine"]    = mine
                known["message"] = message

    def _drop(self, kind: str) -> None:
        with self.lock:
            known = self.active.pop(kind, None)
        if known is not None and known.get("mine"):
            self.logger.muted(f"contention cleared: {kind}")

    def _resolve_device(self):
        target = str(self.paths.repo_root.resolve())
        best   = ("", None)
        try:
            for line in open("/proc/mounts").read().splitlines():
                parts = line.split()
                if len(parts) < 2:
                    continue
                source, mount = parts[0], parts[1]
                if target.startswith(mount) and len(mount) >= len(best[0]) and source.startswith("/dev/"):
                    best = (mount, source.rsplit("/", 1)[-1])
        except OSError:
            return None
        return best[1]
