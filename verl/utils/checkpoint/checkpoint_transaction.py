"""Complete local checkpoints are published only after all files are durable.

Size manifests detect partial/truncated writes, not arbitrary same-size corruption.
Atomic rename requires the staging and final directory on the same filesystem.
"""
import json
import os
from pathlib import Path
import shutil
import uuid


MANIFEST = "CHECKPOINT_COMPLETE.json"
COMPONENT_MANIFEST = "SHARDS_COMPLETE.json"


def fsync_directory(path):
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def inventory(path, *, exclude=()):
    path = Path(path)
    rows = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError(f"Checkpoint cannot contain symlink: {item}")
        if item.is_file() and item.name not in exclude:
            relative = item.relative_to(path).as_posix()
            size = item.stat().st_size
            if size <= 0:
                raise ValueError(f"Empty checkpoint file: {item}")
            rows[relative] = size
    return rows


def sync_files(path, rows):
    # torch shards are already synced by their writers. This also covers HF
    # metadata/tokenizer files written by save_pretrained before publication.
    for relative in rows:
        with (Path(path) / relative).open("rb+") as stream:
            os.fsync(stream.fileno())
    fsync_directory(path)


def _validate_files(path, rows):
    if not rows:
        raise ValueError(f"Checkpoint has empty file manifest: {path}")
    root = Path(path).resolve()
    for relative, size in rows.items():
        item = root / relative
        if item.is_symlink() or not item.resolve().is_relative_to(root):
            raise ValueError(f"Checkpoint path escapes root: {relative}")
        if not item.is_file() or item.stat().st_size != size or size <= 0:
            raise ValueError(f"Missing/truncated checkpoint file: {item}")


def seal_component(path, world_size, step):
    path = Path(path)
    for rank in range(world_size):
        for prefix in ("model", "optim", "extra_state"):
            item = path / f"{prefix}_world_size_{world_size}_rank_{rank}.pt"
            if not item.is_file() or item.stat().st_size <= 0:
                raise ValueError(f"Missing rank checkpoint shard: {item}")
    record = {"version": 1, "world_size": world_size, "global_step": step,
              "files": inventory(path, exclude=(COMPONENT_MANIFEST,))}
    sync_files(path, record["files"])
    atomic_text(path / COMPONENT_MANIFEST, json.dumps(record, indent=2) + "\n")
    return record


def validate_component(path, world_size=None):
    path = Path(path)
    record = json.loads((path / COMPONENT_MANIFEST).read_text(encoding="utf-8"))
    if record.get("version") != 1:
        raise ValueError(f"Unsupported checkpoint manifest: {path}")
    if world_size is not None and record["world_size"] != world_size:
        raise ValueError(f"Sharded checkpoint world size {record['world_size']} != {world_size}; resharding is unsupported")
    _validate_files(path, record["files"])
    for rank in range(record["world_size"]):
        for prefix in ("model", "optim", "extra_state"):
            if f"{prefix}_world_size_{record['world_size']}_rank_{rank}.pt" not in record["files"]:
                raise ValueError("Incomplete per-rank component manifest")
    return record


def begin_checkpoint(root, step):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"global_step_{int(step)}"
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {final}")
    stage = root / f".global_step_{int(step)}.incomplete-{uuid.uuid4().hex}"
    stage.mkdir()
    return stage, final


def publish_checkpoint(stage, final, step, components=("actor",)):
    stage, final = Path(stage).resolve(), Path(final).resolve()
    if stage.parent != final.parent or not stage.name.startswith(".global_step_"):
        raise ValueError("Checkpoint publication requires a sibling staging directory")
    if final.exists():
        raise FileExistsError(f"Refusing to replace checkpoint: {final}")
    for name in components:
        component = validate_component(stage / name)
        if component["global_step"] != step:
            raise ValueError("Checkpoint component step mismatch")
    for name in ("data.pt", "driver_state.pt", "environment_sampler.json"):
        if not (stage / name).is_file():
            raise ValueError(f"Missing complete training state: {name}")
    record = {"version": 1, "global_step": step, "components": list(components),
              "files": inventory(stage, exclude=(MANIFEST,))}
    sync_files(stage, record["files"])
    atomic_text(stage / MANIFEST, json.dumps(record, indent=2) + "\n")
    validate_checkpoint(stage)
    os.rename(stage, final)
    fsync_directory(final.parent)
    atomic_text(final.parent / "latest_checkpointed_iteration.txt", str(step) + "\n")
    return final


def validate_checkpoint(path):
    path = Path(path)
    record = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
    if record.get("version") != 1:
        raise ValueError(f"Unsupported complete checkpoint: {path}")
    if "actor" not in record.get("components", []) or any(name not in ("actor", "critic") for name in record["components"]):
        raise ValueError("Complete checkpoint must identify the actor and optional critic")
    for required in ("data.pt", "driver_state.pt", "environment_sampler.json"):
        if required not in record["files"]:
            raise ValueError(f"Checkpoint manifest omits training state: {required}")
    _validate_files(path, record["files"])
    for name in record["components"]:
        component = validate_component(path / name)
        if component["global_step"] != record["global_step"]:
            raise ValueError("Checkpoint component step mismatch")
    return record


def rotate_complete_checkpoints(root, keep):
    """Delete only old, validated checkpoints inside this exact run root."""
    if keep is None:
        return []
    if not isinstance(keep, int) or keep < 1:
        raise ValueError("Checkpoint retention must be a positive integer or None")
    root = Path(root).resolve()
    complete = []
    for path in root.glob("global_step_*"):
        if path.is_symlink() or not path.is_dir() or path.resolve().parent != root:
            continue
        try:
            record = validate_checkpoint(path)
        except (OSError, ValueError, KeyError):
            continue  # Leave incomplete/legacy/external artifacts untouched.
        complete.append((record["global_step"], path))
    removed = []
    for _, path in sorted(complete)[:-keep]:
        if path.resolve().parent != root:
            raise ValueError("Refusing checkpoint deletion outside run root")
        shutil.rmtree(path)
        removed.append(str(path))
    return removed
