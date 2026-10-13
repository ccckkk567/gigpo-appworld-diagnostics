"""Materialize only the two package LFS objects from the pinned AppWorld source."""
import base64
import csv
import hashlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import sys
import urllib.request

REVISION = '42b5bcf3cd334fee33f0c37c02070a9f5807add5'
# OIDs and sizes read from the LFS pointers at that exact upstream commit.
BUNDLES = {
    'apps.bundle': ('88d21fc526c1655bb3eee4adfca78ccac793921e4506f28f734ecdb19af77a62', 193950),
    'tests.bundle': ('7b93343db5efd81b542e68e68150dd5ea5d59d8dcd64137d41bde4027b235dc9', 204701),
}


def main():
    prefix = Path(os.environ['GIGPO_VENV']).resolve()
    if Path(sys.prefix).resolve() != prefix or sys.prefix == sys.base_prefix:
        raise RuntimeError('Use the dedicated GIGPO venv')
    dist = metadata.distribution('appworld')
    if dist.version != '0.2.0.dev0':
        raise RuntimeError(f'Unexpected AppWorld version: {dist.version}')
    records = [f for f in (dist.files or []) if str(f).endswith('.dist-info/RECORD')]
    if len(records) != 1:
        raise RuntimeError('Expected one AppWorld RECORD')
    record_path = Path(dist.locate_file(records[0])).resolve()
    if not record_path.is_relative_to(prefix):
        raise RuntimeError('AppWorld RECORD is outside the venv')
    rows = list(csv.reader(io.StringIO(record_path.read_text(encoding='utf-8'))))
    report = []
    for name, (oid, size) in BUNDLES.items():
        relative = f'appworld/.source/{name}'
        path = Path(dist.locate_file(relative)).resolve()
        if not path.is_relative_to(prefix):
            raise RuntimeError(f'Bundle is outside the venv: {path}')
        entries = [row for row in rows if row[0] == relative]
        if len(entries) != 1:
            raise RuntimeError(f'Missing or duplicate RECORD entry: {relative}')
        original = path.read_bytes()
        expected_pointer = f'version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {size}\n'.encode()
        downloaded = False
        url = f'https://media.githubusercontent.com/media/StonyBrookNLP/appworld/{REVISION}/src/{relative}'
        if original.replace(b'\r\n', b'\n') == expected_pointer:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read(size + 1)
            if len(data) != size or hashlib.sha256(data).hexdigest() != oid:
                raise RuntimeError(f'LFS size/SHA256 mismatch for {name}; refusing replacement')
            backup = prefix / 'gigpo-appworld-lfs-backups'
            backup.mkdir(exist_ok=True)
            pointer_backup = backup / f'{name}.pointer'
            if not pointer_backup.exists():
                pointer_backup.write_bytes(original)
            record_backup = backup / 'RECORD.original'
            if not record_backup.exists():
                record_backup.write_bytes(record_path.read_bytes())
            temporary = path.with_name(f'{name}.{os.getpid()}.partial')
            temporary.write_bytes(data)
            os.replace(temporary, path)
            downloaded = True
        elif len(original) != size or hashlib.sha256(original).hexdigest() != oid:
            raise RuntimeError(f'Unrecognized bundle contents: {name}; refusing replacement')
        entries[0][1:] = ['sha256=' + base64.urlsafe_b64encode(bytes.fromhex(oid)).rstrip(b'=').decode(), str(size)]
        report.append(dict(file=name, sha256=oid, bytes=size, downloaded=downloaded, source=url))
    stream = io.StringIO(newline='')
    csv.writer(stream, lineterminator='\n').writerows(rows)
    record_path.write_text(stream.getvalue(), encoding='utf-8', newline='\n')
    (Path(os.environ['TMPDIR']) / 'appworld-lfs-bundles.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('APPWORLD_LFS_BUNDLES_READY ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
