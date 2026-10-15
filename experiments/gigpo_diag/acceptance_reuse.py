"""Narrow reviewed exception to commit identity, not a generic preflight bypass.

Only the diagnostic additions reviewed on 2026-10-15 may reuse the successful
58825a7 resource/protocol/checkpoint acceptance. Package/model/RAM/GPU identity
is still checked by bounded_acceptance. This code is a provenance check, not a
security boundary against someone modifying the checker itself.
"""
import hashlib
from pathlib import Path
import subprocess


SOURCE_COMMIT = '58825a74e7de378070970581ae4d8d7b344c5235'
REVIEWED_FILES = {
    # Only observer import/decorator added; native update body is unchanged.
    'verl/workers/fsdp_workers.py': '2c393aa95f170277f612d4be21fdec1e79ae6aa715f50ea7c219c6a93b1a7a02',
    'experiments/gigpo_diag/actor_update_evidence.py': '4447b47cb1134270a2a9bcdb61722f242743982a81104138d2daee08ec089fed',
    # Full schedule and resource protections unchanged; add durable diagnostics.
    'experiments/gigpo_diag/train.sbatch': 'fb7b68eebc1280e1559f1205994283af158310da28b10a3fc93283905fddb80b',
    'experiments/gigpo_diag/bounded_acceptance.py': 'ed74527c318dcc8e3dc5015323eed00cd8da3606c88c97d7a9a0c3f04f6296d1',
}
# These files do not execute in the full-training path, except this checker.
OTHER_REVIEWED_PATHS = {
    'experiments/gigpo_diag/acceptance_reuse.py',
    'experiments/gigpo_diag/test_acceptance_reuse.py',
    'experiments/gigpo_diag/test_full_training_retry.py',
    'experiments/gigpo_diag/test_actor_update_evidence.py',
    'experiments/gigpo_diag/test_actor_verify_job.py',
    'experiments/gigpo_diag/verify_actor.sbatch',
    'docs/APPWORLD_BOUNDED_RUNTIME_IMPLEMENTATION_20261014.md',
}


def validate_changes(paths, read_content):
    unknown = set(paths) - REVIEWED_FILES.keys() - OTHER_REVIEWED_PATHS
    if unknown:
        raise RuntimeError(f'Acceptance reuse refused: unreviewed changes {sorted(unknown)}')
    hashes = {}
    for path in sorted(set(paths) & REVIEWED_FILES.keys()):
        digest = hashlib.sha256(read_content(path)).hexdigest()
        if digest != REVIEWED_FILES[path]:
            raise RuntimeError(f'Acceptance reuse refused: unreviewed content in {path}')
        hashes[path] = digest
    return hashes


def reviewed_changes(source, current):
    if source != SOURCE_COMMIT:
        raise RuntimeError('Acceptance reuse supports only the reviewed 58825a7 baseline')
    repo = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.check_output(['git', *args], cwd=repo)

    if git('rev-parse', 'HEAD').decode().strip() != current:
        raise RuntimeError('Acceptance reuse refused: running commit differs from HEAD')
    if git('status', '--porcelain', '--untracked-files=no').strip():
        raise RuntimeError('Acceptance reuse refused: tracked checkout changes')
    git('merge-base', '--is-ancestor', source, current)
    paths = git('diff', '--name-only', '--no-renames', source, current).decode().splitlines()
    hashes = validate_changes(paths, lambda path: git('show', f'{current}:{path}'))
    return {'source_commit': source, 'code_commit': current,
            'changed_paths': paths, 'reviewed_content_sha256': hashes}
