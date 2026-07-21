from __future__ import annotations

import os
import pwd


class ProcStats:

    PAGE = os.sysconf("SC_PAGE_SIZE")

    @staticmethod
    def username(uid: int) -> str:
        try:
            return pwd.getpwuid(uid).pw_name
        except KeyError:
            return str(uid)

    @staticmethod
    def pss(pid: int) -> int | None:
        try:
            for line in open(f"/proc/{pid}/smaps_rollup"):
                if line.startswith("Pss:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
        return None

    @staticmethod
    def private(pid: int) -> int | None:
        try:
            parts    = open(f"/proc/{pid}/statm").read().split()
            resident = int(parts[1])
            shared   = int(parts[2])
        except (OSError, ValueError, IndexError):
            return None
        return max(0, resident - shared) * ProcStats.PAGE

    @staticmethod
    def attributed(pid: int) -> int | None:
        pss = ProcStats.pss(pid)
        if pss is not None:
            return pss
        return ProcStats.private(pid)

    @staticmethod
    def ppid(pid: int) -> int:
        try:
            stat   = open(f"/proc/{pid}/stat").read()
            fields = stat[stat.rindex(")") + 2 :].split()
            return int(fields[1])
        except (OSError, ValueError, IndexError):
            return 0
