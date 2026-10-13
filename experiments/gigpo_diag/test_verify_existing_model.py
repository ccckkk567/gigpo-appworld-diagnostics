"""Offline model integrity checks using tiny synthetic files, no downloads."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.gigpo_diag.verify_existing_model import MODEL, REVISION, verify_existing_model


class ExistingModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        config = json.dumps(dict(model_type='qwen2', hidden_size=3584, num_hidden_layers=28)).encode()
        self.manifest = dict(model_id=MODEL, revision=REVISION, files=[])
        for name, data in [('config.json', config), ('model-00001-of-00001.safetensors', b'tiny synthetic weight bytes')]:
            (self.root / name).write_bytes(data)
            item = dict(rfilename=name, size=len(data))
            if name.endswith('.safetensors'):
                item['lfs'] = dict(sha256=hashlib.sha256(data).hexdigest())
            else:
                item['blobId'] = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
            self.manifest['files'].append(item)

    def test_valid_files_verified_without_writing_model_directory(self):
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        identity = verify_existing_model(self.root, self.manifest)
        self.assertEqual(identity['revision'], REVISION)
        self.assertEqual(set(identity['files_sha256']), set(before))
        self.assertEqual({p.name: p.read_bytes() for p in self.root.iterdir()}, before)

    def test_same_size_corruption_is_rejected(self):
        file = next(self.root.glob('*.safetensors'))
        file.write_bytes(b'x' * file.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            verify_existing_model(self.root, self.manifest)

    def test_missing_or_extra_weight_shards_are_rejected(self):
        extra = self.root / 'extra.safetensors'
        extra.write_bytes(b'fixture')
        with self.assertRaisesRegex(ValueError, 'weight shards'):
            verify_existing_model(self.root, self.manifest)
        extra.unlink()
        next(self.root.glob('*.safetensors')).unlink()
        with self.assertRaisesRegex(ValueError, 'weight shards'):
            verify_existing_model(self.root, self.manifest)

    def test_wrong_revision_is_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['revision'] = 'unverified'
        with self.assertRaisesRegex(ValueError, 'revision'):
            verify_existing_model(self.root, manifest)

    def test_small_git_blob_and_size_are_checked(self):
        config = self.root / 'config.json'
        data = config.read_bytes().replace(b'3584', b'1536')
        config.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            verify_existing_model(self.root, self.manifest)
        config.write_bytes(data + b' ')
        with self.assertRaisesRegex(ValueError, 'size mismatch'):
            verify_existing_model(self.root, self.manifest)


if __name__ == '__main__':
    unittest.main()
