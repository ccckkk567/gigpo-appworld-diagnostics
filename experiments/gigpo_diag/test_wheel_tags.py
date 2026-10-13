"""Regression checks for real PyPI WHEEL headers observed after Orion job 160510."""
import base64
import csv
from email.parser import Parser
import hashlib
import importlib.metadata as metadata
import io
import json
from pathlib import Path

from packaging.tags import compatible_tags, cpython_tags, parse_tag
import pytest

from experiments.gigpo_diag.repair_wheel_tags import repaired_bytes, repair_distribution

EVIDENCE = json.loads(Path(__file__).with_name('WHEEL_TAG_EVIDENCE.json').read_text())
PLATFORMS = ['manylinux_2_17_x86_64', 'manylinux2014_x86_64', 'manylinux2010_x86_64', 'linux_x86_64']
LINUX312 = set(cpython_tags((3, 12), abis=['cp312'], platforms=PLATFORMS)) | set(compatible_tags((3, 12), interpreter='cp312', platforms=PLATFORMS))


def tags(raw):
    # Same header parser used by pip 25.3 check_unsupported.
    return set().union(*(parse_tag(t) for t in Parser().parsestr(raw).get_all('Tag', [])))


@pytest.mark.parametrize('evidence', EVIDENCE, ids=lambda e: e['package'])
def test_actual_upstream_defect_and_narrow_repair(evidence):
    original = evidence['wheel_metadata'].encode()
    assert tags(original.decode()).isdisjoint(LINUX312)
    fixed = repaired_bytes(original, evidence, LINUX312)
    assert tags(fixed.decode()) & LINUX312
    assert repaired_bytes(fixed, evidence, LINUX312) == fixed
    with pytest.raises(RuntimeError, match='Unrecognized'):
        repaired_bytes(original + b'Unexpected: header\n', evidence, LINUX312)
    with pytest.raises(RuntimeError, match='do not support'):
        repaired_bytes(original, evidence, set(parse_tag('cp312-cp312-win_amd64')))


@pytest.mark.parametrize('evidence', EVIDENCE, ids=lambda e: e['package'])
def test_record_backup_idempotence_and_venv_boundary(tmp_path, evidence):
    prefix = tmp_path / 'venv'
    site = prefix / 'lib' / 'site-packages'
    info = site / f"{evidence['package']}-{evidence['version']}.dist-info"
    info.mkdir(parents=True)
    (info / 'METADATA').write_text(f"Metadata-Version: 2.1\nName: {evidence['package']}\nVersion: {evidence['version']}\n")
    wheel = info / 'WHEEL'
    wheel.write_bytes(evidence['wheel_metadata'].encode())
    record = info / 'RECORD'
    record.write_text(f'{info.name}/WHEEL,,\n{info.name}/RECORD,,\nunchanged.py,sha256=existing,123\n')
    dist = metadata.Distribution.at(info)
    with pytest.raises(RuntimeError, match='outside the venv'):
        repair_distribution(dist, evidence, tmp_path / 'other', LINUX312)
    original_record = record.read_bytes()
    assert repair_distribution(dist, evidence, prefix, LINUX312)['changed']
    rows = list(csv.reader(io.StringIO(record.read_text())))
    assert rows[0][1] == 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(wheel.read_bytes()).digest()).rstrip(b'=').decode()
    assert rows[0][2] == str(wheel.stat().st_size)
    assert rows[2] == ['unchanged.py', 'sha256=existing', '123']
    backup = prefix / 'gigpo-wheel-tag-backups' / f"{evidence['package']}-{evidence['version']}"
    assert (backup / 'WHEEL.original').read_bytes() == evidence['wheel_metadata'].encode()
    assert (backup / 'RECORD.original').read_bytes() == original_record
    assert not repair_distribution(dist, evidence, prefix, LINUX312)['changed']
