from __future__ import annotations

import json
import socket
import threading
import time
from collections import deque
from datetime    import datetime

from project_paths  import ProjectPaths
from system_monitor import SystemMonitor
from web_logger     import WebLogger


class GpuWatchdog:

    INTERVAL    = 4.0
    GRACE_S     = 30.0
    EVENT_LIMIT = 50
    LOG_NAME    = "intrusions.jsonl"

    def __init__(self, system: SystemMonitor, paths: ProjectPaths, logger: WebLogger) -> None:
        self.system    = system
        self.paths     = paths
        self.logger    = logger
        self.host      = socket.gethostname()
        self.lock      = threading.Lock()
        self.armed     = False
        self.incidents = {}
        self.gpus      = []
        self.events    = deque(maxlen=self.EVENT_LIMIT)
        self.count     = 0
        self.critical  = 0
        self.log_path  = self.paths.gpu_guard_dir / self.LOG_NAME

    def start(self) -> None:
        self.paths.gpu_guard_dir.mkdir(parents=True, exist_ok=True)
        worker = threading.Thread(target=self._watch, daemon=True)
        worker.start()
        self.armed = True
        self.logger.muted(f"gpu watchdog armed (interval {self.INTERVAL:.0f}s, journal {self.log_path})")

    def state(self) -> dict:
        with self.lock:
            active = [incident["record"] for incident in self.incidents.values() if incident["status"] != "survived"]
            return {
                "armed"    : self.armed,
                "active"   : active,
                "gpus"     : self.gpus,
                "events"   : list(self.events),
                "count"    : self.count,
                "critical" : self.critical,
                "log_path" : str(self.log_path),
            }

    def _watch(self) -> None:
        while True:
            time.sleep(self.INTERVAL)
            try:
                self._evaluate()
            except Exception as exc:
                self.logger.error(f"gpu watchdog error: {exc}")

    def _evaluate(self) -> None:
        occupancy   = self.system.gpu_occupancy()
        server_gpus = self._server_gpus(occupancy)
        intruders   = self._detect(occupancy)

        self._register(intruders, server_gpus)
        self._escalate(intruders, server_gpus)

        with self.lock:
            self.gpus = server_gpus

    def _server_gpus(self, occupancy: list[dict]) -> list[dict]:
        summary = []
        for device in occupancy:
            owners = {proc["owner"] for proc in device["procs"] if proc["owner"]}
            summary.append({
                "index"     : device["index"],
                "name"      : device["name"],
                "util"      : device["util"],
                "mem_used"  : device["mem_used"],
                "mem_total" : device["mem_total"],
                "holders"   : [{"user": proc["owner"], "pid": proc["pid"], "mem_mib": proc["mem"], "cmd": proc["cmd"]} for proc in device["procs"]],
                "mine"      : self.system.user in owners,
                "others"    : bool(owners - {self.system.user}),
                "free"      : len(device["procs"]) == 0,
            })
        return summary

    def _detect(self, occupancy: list[dict]) -> dict:
        intruders = {}
        for device in occupancy:
            mine = [proc for proc in device["procs"] if proc["owner"] == self.system.user]
            if not mine:
                continue

            for proc in device["procs"]:
                owner = proc["owner"]
                if owner is None or owner == self.system.user:
                    continue
                intruders[(device["uuid"], proc["pid"])] = self._intrusion(device, proc, mine)
        return intruders

    def _intrusion(self, device: dict, proc: dict, mine: list[dict]) -> dict:
        return {
            "gpu_index" : device["index"],
            "gpu_uuid"  : device["uuid"],
            "gpu_name"  : device["name"],
            "user"      : proc["owner"],
            "pid"       : proc["pid"],
            "mem_mib"   : proc["mem"],
            "cmd"       : proc["cmd"],
            "mine_pids" : [m["pid"] for m in mine],
            "mine_mib"  : round(sum(m["mem"] for m in mine), 1),
            "util"      : device["util"],
            "mem_used"  : device["mem_used"],
            "mem_total" : device["mem_total"],
            "status"    : "active",
            "since"     : datetime.now().isoformat(timespec="seconds"),
        }

    def _register(self, intruders: dict, server_gpus: list[dict]) -> None:
        now = time.monotonic()

        for key, record in intruders.items():
            incident = self.incidents.get(key)
            if incident is None:
                self.incidents[key] = {"record": record, "mine_pids": list(record["mine_pids"]), "status": "active", "last_seen": now}
                self._persist(record, server_gpus, "intrusion")
                self._raise(record, "intrusion")
            else:
                incident["last_seen"] = now

    def _escalate(self, intruders: dict, server_gpus: list[dict]) -> None:
        now = time.monotonic()

        for key, incident in self.incidents.items():
            if incident["status"] != "active":
                continue

            dead = [pid for pid in incident["mine_pids"] if self.system.pid_owner(pid) != self.system.user]

            if dead:
                self._flag_critical(incident, dead, server_gpus)
            elif key not in intruders and now - incident["last_seen"] > self.GRACE_S:
                incident["status"]           = "survived"
                incident["record"]["status"] = "survived"

    def _flag_critical(self, incident: dict, dead: list[int], server_gpus: list[dict]) -> None:
        record                = incident["record"]
        incident["status"]    = "critical"
        record["status"]      = "critical"
        record["dead_pids"]   = dead
        record["critical_at"] = datetime.now().isoformat(timespec="seconds")

        self._persist(record, server_gpus, "critical")

        with self.lock:
            self.critical += 1
            self.events.append(dict(record))

        self.logger.error(f"CRITICAL: your pid(s) {dead} on gpu {record['gpu_index']} [{record['gpu_name']}] DIED after {record['user']} (pid {record['pid']}) invaded it")

    def _persist(self, record: dict, server_gpus: list[dict], kind: str) -> None:
        entry = {
            "kind"        : kind,
            "detected_at" : record["since"],
            "host"        : self.host,
            "user_me"     : self.system.user,
            "gpu"         : {"index": record["gpu_index"], "uuid": record["gpu_uuid"], "name": record["gpu_name"], "util": record["util"], "mem_used": record["mem_used"], "mem_total": record["mem_total"]},
            "intruder"    : {"user": record["user"], "pid": record["pid"], "mem_mib": record["mem_mib"], "cmd": record["cmd"]},
            "mine"        : {"pids": record["mine_pids"], "mem_mib": record["mine_mib"]},
            "all_gpus"    : server_gpus,
        }

        if kind == "critical":
            entry["dead_pids"]   = record.get("dead_pids", [])
            entry["critical_at"] = record.get("critical_at")

        try:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        except OSError as exc:
            self.logger.error(f"gpu watchdog could not write journal: {exc}")

    def _raise(self, record: dict, kind: str) -> None:
        with self.lock:
            self.count += 1
            self.events.append(dict(record))

        self.logger.error(f"GPU INTRUSION: {record['user']} (pid {record['pid']}, {round(record['mem_mib'])} MiB) joined gpu {record['gpu_index']} [{record['gpu_name']}] while you hold pids {record['mine_pids']}")
