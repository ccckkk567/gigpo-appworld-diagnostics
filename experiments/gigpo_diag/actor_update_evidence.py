"""Opt-in worker evidence, written before Ray transports an exception.

This observes the unchanged native update; it never retries, substitutes an
exception, disables compilation, or changes an optimizer result.
"""
import functools
import gzip
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import traceback


def _json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def _event(root, row):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"rank-{row['rank']}-pid-{row['pid']}.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def record_actor_update(function):
    @functools.wraps(function)
    def wrapped(self, data, *args, **kwargs):
        if os.environ.get("GIGPO_ACTOR_VERIFY") != "1":
            return function(self, data, *args, **kwargs)
        root = Path(os.environ["GIGPO_RUN_DIR"]) / "actor-update"
        row = {"rank": int(self.rank), "pid": os.getpid(), "host": socket.gethostname(),
               "update": data.meta_info.get("diag_update_target"),
               "code_commit": os.environ.get("GIGPO_CODE_COMMIT")}
        try:
            _event(root, {**row, "event": "begin", "time_ns": time.time_ns(),
                         "use_torch_compile": self.config.actor.get("use_torch_compile", True),
                         "entropy_coeff": self.config.actor.entropy_coeff,
                         "use_remove_padding": self.config.model.use_remove_padding,
                         "inputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                                    for k, v in data.batch.items()}})
        except Exception as exc:
            print(f"ACTOR_EVIDENCE_WRITE_FAILED begin: {exc}", file=sys.stderr, flush=True)
        try:
            result = function(self, data, *args, **kwargs)
        except BaseException as exc:
            raw = traceback.format_exc()
            inner = getattr(exc, "inner_exception", None)
            if isinstance(inner, BaseException):
                raw += "\nExplicit inner_exception:\n" + "".join(
                    traceback.format_exception(type(inner), inner, inner.__traceback__))
            try:
                root.mkdir(parents=True, exist_ok=True)
                path = root / f"rank-{row['rank']}-pid-{row['pid']}-{time.time_ns()}.traceback.txt"
                with path.open("x", encoding="utf-8") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                _event(root, {**row, "event": "failed", "time_ns": time.time_ns(),
                             "error_type": type(exc).__name__, "traceback": str(path)})
                print(f"ACTOR_UPDATE_TRACEBACK {path}", file=sys.stderr, flush=True)
            except Exception as write_error:
                print(f"ACTOR_EVIDENCE_WRITE_FAILED exception: {write_error}", file=sys.stderr, flush=True)
            print(raw, file=sys.stderr, flush=True)
            raise
        try:
            values = result.meta_info["metrics"]["actor/grad_norm"]
            if not isinstance(values, (list, tuple)):
                values = [values]
            norms = [float(x) for x in values]
            _event(root, {**row, "event": "returned", "time_ns": time.time_ns(),
                         "finite_grad_norms": bool(norms) and all(math.isfinite(x) for x in norms),
                         "grad_norms": [x if math.isfinite(x) else None for x in norms]})
        except Exception as exc:
            print(f"ACTOR_EVIDENCE_WRITE_FAILED return: {exc}", file=sys.stderr, flush=True)
        return result
    return wrapped


def verify(root, expected_world_size=4):
    """Require native update AND optimizer evidence from every requested worker."""
    if expected_world_size not in (2, 4):
        raise ValueError("Actor verification supports two or four GPU workers")
    root = Path(root)
    rows = [json.loads(line) for path in (root / "actor-update").glob("rank-*.jsonl")
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(row["event"] == "failed" for row in rows):
        raise RuntimeError("Actor worker failure recorded; inspect actor-update/*.traceback.txt")
    begins = [row for row in rows if row["event"] == "begin"]
    returns = [row for row in rows if row["event"] == "returned"]
    if (len(begins) != expected_world_size or len(returns) != expected_world_size or
            {row["rank"] for row in begins} != set(range(expected_world_size)) or
            {row["rank"] for row in returns} != set(range(expected_world_size)) or
            len({row["pid"] for row in returns}) != expected_world_size):
        raise RuntimeError(f"Need exactly one native update on each of {expected_world_size} distinct GPU workers")
    for begin in begins:
        returned = next(row for row in returns if row["rank"] == begin["rank"])
        if (begin["pid"] != returned["pid"] or begin["update"] != 1 or returned["update"] != 1 or
                begin["time_ns"] >= returned["time_ns"] or
                not begin["use_torch_compile"] or not begin["use_remove_padding"] or
                begin["entropy_coeff"] != 0.001 or not returned["finite_grad_norms"]):
            raise RuntimeError(f"Incomplete compiled actor update evidence for rank {begin['rank']}")
        optimizer_path = root / "diagnostics" / f"optimizer.pid-{begin['pid']}.jsonl.gz"
        with gzip.open(optimizer_path, "rt", encoding="utf-8") as stream:
            optimizer = [json.loads(line) for line in stream if line.strip()]
        if not optimizer or not any(row.get("optimizer_step_local", 0) >= 1 for row in optimizer):
            raise RuntimeError(f"No native optimizer step for rank {begin['rank']}")
    updates = []
    for path in (root / "diagnostics").glob("updates.pid-*.jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            updates.extend(json.loads(line) for line in stream if line.strip())
    if len(updates) != 1 or updates[0].get("policy_update") != 1 or not updates[0].get("native_actor_update_returned"):
        raise RuntimeError("Driver must confirm exactly one completed policy update")
    result = {"passed": True, "world_size": expected_world_size,
              "scope": "one real GiGPO update, not full training acceptance",
              "code_commit": os.environ.get("GIGPO_CODE_COMMIT"), "workers": returns}
    _json_write(root / "ACTOR_UPDATE_VERIFIED.json", result)
    print("ACTOR_UPDATE_VERIFIED", root, flush=True)


def environment(root):
    packages = {}
    for package in ("torch", "triton", "ray", "vllm", "flash-attn", "transformers"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    # Explicit allowlist; never dump credentials or the entire job environment.
    names = ("CUDA_VISIBLE_DEVICES", "GIGPO_NUM_GPUS", "TMPDIR", "RAY_TMPDIR", "XDG_CACHE_HOME",
             "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCH_COMPILE_DISABLE",
             "TORCHDYNAMO_DISABLE", "TORCH_LOGS", "TORCHDYNAMO_VERBOSE", "CC", "CXX")
    _json_write(Path(root) / "compiler_environment.json",
                {"host": socket.gethostname(), "python": sys.version, "executable": sys.executable,
                 "code_commit": os.environ.get("GIGPO_CODE_COMMIT"), "packages": packages,
                 "environment": {name: os.environ.get(name) for name in names}})


if __name__ == "__main__":
    if sys.argv[1] == "environment":
        environment(sys.argv[2])
    elif sys.argv[1] == "verify":
        verify(sys.argv[2], int(os.environ["GIGPO_NUM_GPUS"]))
    else:
        raise ValueError("Expected environment or verify")
