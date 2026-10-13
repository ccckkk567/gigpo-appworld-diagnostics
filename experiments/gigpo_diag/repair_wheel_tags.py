"""Repair two verified upstream WHEEL metadata defects, never library code.

See WHEEL_TAG_EVIDENCE.json and CONFIG_DIFF.md. A normal pip check still follows.
"""
import base64
import csv
import hashlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import sys

from packaging.tags import parse_tag, sys_tags


def repaired_bytes(original, evidence, supported):
    expected = evidence['wheel_metadata'].encode('utf-8')
    fixed = evidence['repaired_wheel_metadata'].encode('utf-8')
    if original not in (expected, fixed):
        raise RuntimeError(f"Unrecognized WHEEL metadata for {evidence['package']}; refusing repair")
    target_tags = set()
    for line in fixed.decode().splitlines():
        if line.startswith('Tag: '):
            target_tags.update(parse_tag(line[5:]))
    if target_tags.isdisjoint(supported):
        raise RuntimeError(f"Upstream filename tags do not support this interpreter: {evidence['package']}")
    return fixed


def repair_distribution(dist, evidence, prefix, supported):
    name = evidence['package']
    if dist.version != evidence['version']:
        raise RuntimeError(f"Unexpected {name} version {dist.version}; refusing repair")
    entries = [p for p in (dist.files or []) if str(p).endswith('.dist-info/WHEEL')]
    if len(entries) != 1:
        raise RuntimeError(f"Expected one WHEEL file for {name}")
    entry = entries[0]
    wheel_path = Path(dist.locate_file(entry)).resolve()
    if not wheel_path.is_relative_to(prefix.resolve()):
        raise RuntimeError(f"Refusing to modify a distribution outside the venv: {wheel_path}")
    original = wheel_path.read_bytes()
    fixed = repaired_bytes(original, evidence, supported)
    record_path = wheel_path.with_name('RECORD')
    original_record = record_path.read_bytes()
    rows = list(csv.reader(io.StringIO(original_record.decode('utf-8'))))
    targets = [row for row in rows if row[0] == str(entry).replace('\\', '/')]
    if len(targets) != 1:
        raise RuntimeError(f"Expected one WHEEL entry in {name} RECORD")
    digest = 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(fixed).digest()).rstrip(b'=').decode()
    changed = original != fixed or targets[0][1:] != [digest, str(len(fixed))]
    if changed:
        backup = prefix / 'gigpo-wheel-tag-backups' / f"{name}-{dist.version}"
        backup.mkdir(parents=True, exist_ok=True)
        for filename, content in [('WHEEL.original', original), ('RECORD.original', original_record)]:
            path = backup / filename
            if not path.exists():
                path.write_bytes(content)
        targets[0][1:] = [digest, str(len(fixed))]
        stream = io.StringIO(newline='')
        csv.writer(stream, lineterminator='\n').writerows(rows)
        wheel_path.write_bytes(fixed)
        record_path.write_bytes(stream.getvalue().encode('utf-8'))
    result = dict(package=name, version=dist.version, changed=changed, wheel_path=str(wheel_path),
                  original_sha256=hashlib.sha256(original).hexdigest(), repaired_sha256=hashlib.sha256(fixed).hexdigest())
    print('WHEEL_TAG_REPAIR ' + json.dumps(result), flush=True)
    return result


def main():
    prefix = Path(os.environ['GIGPO_VENV']).resolve()
    if Path(sys.prefix).resolve() != prefix or sys.prefix == sys.base_prefix:
        raise RuntimeError('Run with the dedicated GIGPO virtual environment')
    if sys.platform != 'linux' or sys.version_info[:2] != (3, 12):
        raise RuntimeError('This repair is scoped to the Orion Linux/Python 3.12 environment')
    evidence = json.loads(Path(__file__).with_name('WHEEL_TAG_EVIDENCE.json').read_text(encoding='utf-8'))
    supported = set(sys_tags())
    results = [repair_distribution(metadata.distribution(e['package']), e, prefix, supported) for e in evidence]
    (Path(os.environ['TMPDIR']) / 'wheel-tag-repair.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
