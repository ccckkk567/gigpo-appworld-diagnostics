"""Local acceptance evidence gate, not a cryptographic signature or security boundary."""
import importlib.metadata
import os
from pathlib import Path
import sys

from experiments.gigpo_diag.appworld_resource import atomic_json, read_json, load_profile


def identity():
    return {'code_commit': os.environ['GIGPO_CODE_COMMIT'],
            'model_path': os.environ['GIGPO_MODEL_PATH'],
            'packages': {p: importlib.metadata.version(p) for p in
                         ('torch', 'vllm', 'triton', 'appworld', 'ipython', 'libcst', 'transformers', 'torchdata')},
            'python_version': sys.version.split()[0], 'world_size': 4,
            'job_requested_mib': int(os.environ['SLURM_MEM_PER_NODE'])}


def verify_compatible(accepted, current, reuse_from=None):
    if not accepted.get('passed') or any(
            accepted.get(k) != v for k, v in current.items() if k != 'code_commit'):
        raise RuntimeError('Missing or incompatible completed preflight; training refused')
    if accepted.get('code_commit') == current['code_commit']:
        return None
    if not reuse_from or reuse_from != accepted.get('code_commit'):
        raise RuntimeError('Preflight code differs; explicit reviewed reuse is required')
    from experiments.gigpo_diag.acceptance_reuse import reviewed_changes
    return reviewed_changes(reuse_from, current['code_commit'])


def main():
    if sys.argv[1] == 'publish':
        root = Path(os.environ['GIGPO_RUN_DIR'])
        profile = load_profile()
        # This command is reached by set -e only after the GPU verifier succeeds.
        # Require its explicit evidence as well, to reject empty directories.
        evidence = root / 'checkpoint-roundtrip' / 'CHECKPOINT_ROUNDTRIP_PASSED.json'
        rows = read_json(evidence, {}).get('ranks', [])
        if (len(rows) != 4 or {r.get('rank') for r in rows} != set(range(4)) or
                not all(r.get('next_update_match') and r.get('dataloader_match') and
                        r.get('cpu_cuda_numpy_python_rng_match') for r in rows)):
            raise RuntimeError(f'GPU roundtrip pass evidence missing: {evidence}')
        live = read_json(root / 'live-protocol' / 'LIVE_PROTOCOL_PASSED.json', {})
        if not live.get('passed'):
            raise RuntimeError('Real HTTP service group acceptance evidence missing')
        atomic_json(root / 'acceptance.json', {'passed': True, **identity(),
                    'resource_profile': profile, 'checkpoint_evidence': str(evidence),
                    'preflight_job_id': os.environ.get('SLURM_JOB_ID')})
        print('BOUNDED_ACCEPTANCE_PUBLISHED', root, flush=True)
    elif sys.argv[1] == 'verify':
        root = Path(os.environ['GIGPO_ACCEPTANCE_DIR'])
        accepted = read_json(root / 'acceptance.json', {})
        reuse = verify_compatible(accepted, identity(), os.environ.get('GIGPO_ACCEPTANCE_REUSE_FROM'))
        if reuse is not None:
            atomic_json(Path(os.environ['GIGPO_RUN_DIR']) / 'acceptance_reuse.json',
                        {**reuse, 'acceptance_dir': str(root),
                         'scope': 'Existing resource/protocol/checkpoint acceptance only; '
                                  'does not certify compiled actor updates or full training'})
            print('BOUNDED_ACCEPTANCE_REUSED_DIAGNOSTICS_ONLY', reuse, flush=True)
        print('BOUNDED_ACCEPTANCE_VERIFIED', root, flush=True)
    else:
        raise ValueError('Expected publish or verify')


if __name__ == '__main__':
    main()
