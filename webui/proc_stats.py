from __future__ import annotations


class ProcStats:

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
    def ppid(pid: int) -> int:
        try:
            stat   = open(f"/proc/{pid}/stat").read()
            fields = stat[stat.rindex(")") + 2 :].split()
            return int(fields[1])
        except (OSError, ValueError, IndexError):
            return 0
