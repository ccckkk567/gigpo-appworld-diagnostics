"""Post-hoc scalar upload in a separate SwanLab >=0.9 environment.

Read a FINISHED run; never import the trainer or change its dependencies/data.
Without --upload this only prepares a reviewable, hashed local snapshot.
"""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path


def read_snapshot(root):
    manifest_bytes = (root / 'RUN_MANIFEST.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('exit_code') is None:
        raise ValueError('Run has no final exit_code; wait for the training job to end.')
    run_id = manifest['run_id']
    paths = sorted((root / 'diagnostics').glob('metrics.pid-*.jsonl.gz'))
    if not paths:
        raise ValueError('No scalar metric files; nothing to upload.')
    sources = {'RUN_MANIFEST.json': hashlib.sha256(manifest_bytes).hexdigest()}
    merged, omitted = {}, []
    for path in paths:
        raw = path.read_bytes()
        sources[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        # Decode a complete immutable byte snapshot; truncated gzip is an error.
        for line_number, line in enumerate(gzip.decompress(raw).decode('utf-8').splitlines(), 1):
            row = json.loads(line)
            if row.get('run_id') != run_id or row.get('code_commit') != manifest['code_commit']:
                raise ValueError(f'Mixed run/commit at {path.name}:{line_number}')
            step = row['global_step']
            if type(step) is not int or step < 0:
                raise ValueError(f'Invalid global_step at {path.name}:{line_number}')
            values = merged.setdefault(step, {})
            for key, value in row['metrics'].items():
                if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
                    omitted.append({'step': step, 'metric': key, 'reason': 'unknown_or_non_scalar'})
                    continue  # Unknown is never converted to zero.
                if key in values and values[key] != value:
                    raise ValueError(f'Conflicting metric {key!r} at step {step}; refusing to average.')
                values[key] = value
    points = [{'step': step, 'metrics': values} for step, values in sorted(merged.items()) if values]
    if not points:
        raise ValueError('No finite scalar metrics; nothing to upload.')
    info = {key: manifest.get(key) for key in
            ('run_id', 'code_commit', 'job_id', 'env', 'status', 'training_updates', 'policy_updated')}
    return {'schema': 'gigpo-swanlab-backfill-v1', 'source': info,
            'source_sha256': sources, 'points': points, 'omitted': omitted,
            'scope': 'Recorded scalar metrics only; trajectories and selected text samples remain in source logs.'}


def persist_snapshot(payload, output):
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'snapshot.json'
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if target.exists():
        if target.read_text(encoding='utf-8') != text:
            raise ValueError('Output contains a different snapshot; use a new output directory.')
    else:
        target.write_text(text, encoding='utf-8')
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def upload(payload, output, snapshot_hash, project, workspace):
    # Check before importing SDK. Do NOT upgrade the training venv to meet this.
    from packaging.version import Version
    version = importlib.metadata.version('swanlab')
    if Version(version) < Version('0.9.0'):
        raise RuntimeError('Use a separate SwanLab >=0.9 interpreter (existing pilot venv: 0.10.2).')
    import swanlab
    receipt_path = output / 'upload_receipt.json'
    identity = {'snapshot_sha256': snapshot_hash, 'project': project, 'workspace': workspace}
    # Do not silently create another run after an uncertain upload.
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text(encoding='utf-8'))
        if any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError('Upload destination/snapshot differs from existing receipt.')
        if previous['state'] == 'sdk_finish_returned':
            return previous
        raise RuntimeError('Previous upload is incomplete/uncertain; inspect receipt and SwanLab before retrying.')
    receipt = {**identity, 'swanlab_version': version, 'state': 'starting',
               'cloud_contents_verified': False, 'url': None}
    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    save()
    initialized = False
    try:
        run = swanlab.init(
            project=project, workspace=workspace,
            experiment_name=payload['source']['run_id'] + '-backfill',
            description='Post-hoc copy of recorded scalar metrics. No training or evaluation executed by this uploader.',
            config={'backfill': True, 'source': payload['source'], 'snapshot_sha256': snapshot_hash,
                    'scope': payload['scope']},
            logdir=str(output / 'swanlog'), mode='online')
        initialized = True
        receipt.update(state='uploading', url=getattr(run, 'url', None))
        save()
        for point in payload['points']:
            swanlab.log(point['metrics'], step=point['step'])
        swanlab.finish()
        initialized = False
        receipt.update(state='sdk_finish_returned', steps=len(payload['points']))
        save()
        return receipt
    except Exception as exc:
        receipt.update(state='failed_or_uncertain', error_type=type(exc).__name__)
        save()
        raise
    finally:
        if initialized:
            try:
                swanlab.finish()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='Separate new directory for this export')
    parser.add_argument('--project', default='gigpo-training-diagnostics')
    parser.add_argument('--workspace', default='KANSEI')
    parser.add_argument('--upload', action='store_true', help='Upload using this interpreter\'s modern SwanLab SDK')
    args = parser.parse_args()
    root, output = args.run_dir.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        parser.error('--output must be outside the original run directory')
    payload = read_snapshot(root)
    digest = persist_snapshot(payload, output)
    print(json.dumps({'state': 'snapshot_prepared', 'run_id': payload['source']['run_id'],
                      'steps': [p['step'] for p in payload['points']], 'omitted': len(payload['omitted']),
                      'snapshot': str(output / 'snapshot.json')}, ensure_ascii=False))
    if args.upload:
        print(json.dumps(upload(payload, output, digest, args.project, args.workspace)))


if __name__ == '__main__':
    main()
