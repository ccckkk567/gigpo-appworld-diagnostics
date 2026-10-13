"""Offline verification of the user's existing pinned 7B model; read-only."""
import hashlib
import json
from pathlib import Path

MODEL = 'Qwen/Qwen2.5-7B-Instruct'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'


def verify_existing_model(path, manifest=None):
    path = Path(path)
    if manifest is None:
        manifest = json.loads(Path(__file__).with_name('model_files_7b.json').read_text())
    if manifest['model_id'] != MODEL or manifest['revision'] != REVISION:
        raise ValueError('Unexpected source model/revision in checksum manifest')
    expected_shards = {x['rfilename'] for x in manifest['files'] if x['rfilename'].endswith('.safetensors')}
    if {p.name for p in path.glob('*.safetensors')} != expected_shards:
        raise ValueError('Missing or unexpected model weight shards')
    hashes = {}
    for item in manifest['files']:
        name = item['rfilename']
        file = path / name
        size = file.stat().st_size
        if size != item['size']:
            raise ValueError(f'Model file size mismatch: {name}')
        digest = hashlib.sha256()
        blob = hashlib.sha1(f'blob {size}\0'.encode()) if 'lfs' not in item else None
        with file.open('rb') as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
                if blob is not None:
                    blob.update(chunk)
        hashes[name] = digest.hexdigest()
        matched = hashes[name] == item['lfs']['sha256'] if 'lfs' in item else blob.hexdigest() == item['blobId']
        if not matched:
            raise ValueError(f'Model file checksum mismatch: {name}')
    config = json.loads((path / 'config.json').read_text())
    if (config.get('model_type'), config.get('hidden_size'), config.get('num_hidden_layers')) != ('qwen2', 3584, 28):
        raise ValueError('Expected Qwen2.5-7B model architecture')
    return {'model_id': MODEL, 'revision': REVISION, 'tokenizer_revision': REVISION,
            'files_sha256': hashes, 'verification': 'offline official Git-blob and LFS checksums',
            'source_manifest': 'model_files_7b.json'}
