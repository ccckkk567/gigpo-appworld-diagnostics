"""Small, best-effort checkpoint stage measurements; missing fields stay null."""
import json
import os
from pathlib import Path
import time


def _read(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _number(path):
    value = _read(path)
    return int(value) if value and value.isdigit() else value


def memory_snapshot():
    process = {"rss_bytes": None, "vms_bytes": None, "rss_high_water_bytes": None}
    status = _read("/proc/self/status")
    if status is not None:
        names = {"VmRSS": "rss_bytes", "VmSize": "vms_bytes", "VmHWM": "rss_high_water_bytes"}
        for line in status.splitlines():
            key, _, value = line.partition(":")
            if key in names:
                process[names[key]] = int(value.split()[0]) * 1024
    else:
        try:
            import psutil
            info = psutil.Process().memory_info()
            process.update(rss_bytes=info.rss, vms_bytes=info.vms)
        except (ImportError, OSError):
            pass
    cgroups = []
    memberships = _read("/proc/self/cgroup") or ""
    for line in memberships.splitlines():
        hierarchy, controllers, relative = line.split(":", 2)
        if hierarchy == "0" and not controllers:
            base = Path("/sys/fs/cgroup")
            current = base / relative.lstrip("/")
            while current.is_relative_to(base):
                if (current / "memory.current").exists():
                    row = {"path": str(current), "current_bytes": _number(current / "memory.current"),
                           "peak_bytes": _number(current / "memory.peak"), "max_bytes": _number(current / "memory.max")}
                    for filename in ("memory.events", "memory.stat"):
                        raw = _read(current / filename)
                        row[filename] = dict((k, int(v)) for k, v in (x.split() for x in raw.splitlines())) if raw else None
                    cgroups.append(row)
                if current == base:
                    break
                current = current.parent
        elif "memory" in controllers.split(","):
            base = Path("/sys/fs/cgroup/memory")
            current = base / relative.lstrip("/")
            while current.is_relative_to(base):
                if (current / "memory.usage_in_bytes").exists():
                    cgroups.append({"path": str(current), "current_bytes": _number(current / "memory.usage_in_bytes"),
                                    "peak_bytes": _number(current / "memory.max_usage_in_bytes"),
                                    "max_bytes": _number(current / "memory.limit_in_bytes"),
                                    "failcnt": _number(current / "memory.failcnt"),
                                    "memory.stat": _read(current / "memory.stat")})
                if current == base:
                    break
                current = current.parent
    return {"process": process, "cgroups": cgroups}


def tensor_storage_summary(value):
    """Count unique local tensor storages by device; no tensor transfers/copies."""
    seen, totals = set(), {}
    def visit(item):
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        elif hasattr(item, "local_shards"):
            for shard in item.local_shards():
                visit(shard.tensor)
        elif hasattr(item, "to_local"):
            visit(item.to_local())
        elif hasattr(item, "untyped_storage"):
            storage = item.untyped_storage()
            key = (str(item.device), storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key)
                totals[str(item.device)] = totals.get(str(item.device), 0) + storage.nbytes()
    visit(value)
    return totals


def record_checkpoint_stage(stage, rank, checkpoint_path, state=None):
    try:
        row = {"time_unix": time.time(), "pid": os.getpid(), "rank": rank,
               "stage": stage, "checkpoint": str(checkpoint_path), **memory_snapshot()}
        if state is not None:
            row["local_state_storage_bytes_by_device"] = tensor_storage_summary(state)
        line = json.dumps(row, separators=(",", ":"))
        print("CHECKPOINT_MEMORY " + line, flush=True)
        directory = os.environ.get("GIGPO_DIAG_DIR")
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
            with (Path(directory) / f"checkpoint-memory-rank-{rank}-pid-{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
    except Exception as error:
        print(f"Checkpoint memory measurement unavailable: {type(error).__name__}", flush=True)
