"""Run identity and pinned input preparation; no optimizer or policy logic."""
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare_model():
    from huggingface_hub import snapshot_download
    path = Path(os.environ['GIGPO_MODEL_PATH'])
    snapshot_download(MODEL, revision=REVISION, local_dir=str(path),
                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    files = {p.name: sha(p) for p in sorted(path.iterdir()) if p.is_file() and p.name != 'gigpo_model_identity.json' and p.suffix in ('.json', '.safetensors', '.txt', '.model', '.jinja')}
    (path/'gigpo_model_identity.json').write_text(json.dumps({'model_id': MODEL, 'revision': REVISION,
        'tokenizer_revision': REVISION, 'files_sha256': files}, indent=2), encoding='utf-8')
    print('PINNED_MODEL_READY', path, REVISION, flush=True)


def record_provenance(config):
    from omegaconf import OmegaConf
    import verl
    root = Path(__file__).resolve().parents[2]
    actual = Path(verl.__file__).resolve()
    if not actual.is_relative_to(root):
        raise RuntimeError(f'Wrong editable verl: {actual}; expected {root}')
    output = Path(os.environ['GIGPO_RUN_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    model = Path(config.actor_rollout_ref.model.path).resolve()
    identity = json.loads((model/'gigpo_model_identity.json').read_text())
    assert identity['model_id'] == MODEL and identity['revision'] == REVISION
    assert identity['tokenizer_revision'] == REVISION
    model_cfg = json.loads((model/'config.json').read_text())
    assert model_cfg['hidden_size'] == 1536 and model_cfg['num_hidden_layers'] == 28, 'Expected 1.5B identity'
    # Hash all pinned checkpoint/tokenizer files before model loading.
    for name, expected in identity['files_sha256'].items():
        assert sha(model/name) == expected, f'Model/tokenizer file changed: {name}'
    OmegaConf.save(config, output/'resolved_config.yaml', resolve=True)
    versions = {}
    for package in ('torch','transformers','vllm','ray','tensordict','flash-attn','swanlab','scienceworld','appworld','alfworld'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    provenance = {'python':sys.version, 'executable':sys.executable, 'verl_file':str(actual), 'packages':versions,
                  'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'), 'job_id':os.environ.get('SLURM_JOB_ID')}
    if os.environ['GIGPO_ENV'] == 'scienceworld':
        import scienceworld
        provenance['scienceworld_files'] = {str(p): sha(p) for p in Path(scienceworld.__file__).parent.rglob('*.jar')}
        pool = root/'agent_system/environments/env_package/sciworld/variations_idx/L0_idx.json'
        provenance['task_pool_sha256'] = sha(pool)
        provenance['simplifications_preset'] = config.env.sciworld.get('simplifications_preset', 'easy')
    for name, value in [('environment_provenance.json', provenance), ('model_identity.json', identity)]:
        (output/name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    source = json.loads((Path(__file__).parent/'SOURCE.json').read_text())
    manifest = {**source, 'run_id': os.environ['GIGPO_RUN_ID'], 'code_commit':os.environ['GIGPO_CODE_COMMIT'],
                'branch':subprocess.check_output(['git','branch','--show-current'], text=True).strip(),
                'status':'initialized_not_yet_updated', 'policy_updated':False, 'training_updates':0,
                'job_id':os.environ.get('SLURM_JOB_ID'), 'checkpoint_dir':config.trainer.default_local_dir,
                'initial_checkpoint':str(model), 'tracker_requested':'swanlab', 'seed':config.env.seed,
                'resume_from_path':config.trainer.get('resume_from_path'),
                'reference_schedule_epochs':config.trainer.total_epochs, 'formal_holdout_details_saved':False}
    (output/'RUN_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    snapshot = output/'implementation_snapshot'
    for name in ('gigpo/core_gigpo.py','agent_system/environments/env_manager.py','agent_system/multi_turn_rollout/rollout_loop.py','verl/trainer/ppo/ray_trainer.py','experiments/gigpo_diag/diagnostics.py','experiments/gigpo_diag/SOURCE.json','experiments/gigpo_diag/CONFIG_DIFF.md'):
        dest=snapshot/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name,dest)
    print('RUN_IDENTITY_VERIFIED', manifest['run_id'], manifest['code_commit'], actual, flush=True)


def science_reset_check():
    """Four live simulators, two groups, two resets; no policy calls."""
    from agent_system.environments.env_package.sciworld.envs import build_sciworld_envs
    from experiments.gigpo_diag.diagnostics import plain, verify_groups
    pool_path=Path('agent_system/environments/env_package/sciworld/variations_idx/L0_idx.json')
    pool=json.loads(pool_path.read_text())['train']
    env=build_sciworld_envs(seed=0, env_num=2, group_n=2, variations_idx=pool, simplifications_preset='easy')
    results=[]
    try:
        for round_index in range(2):
            obs, infos=env.reset()
            verify_groups({'anchor':obs}, infos, [0,0,1,1])
            for i, info in enumerate(infos):
                results.append({'reset_round':round_index,'group_slot':i//2,'worker_slot':i,
                                'observation':obs[i],'info':plain(info)})
    finally:
        env.close()
    path=Path(os.environ['GIGPO_RUN_DIR'])/'science_group_reset_check.json'
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('LIVE_GROUP_RESET_CHECK_PASSED groups=2 repeats=2 trajectories=0', flush=True)


if __name__ == '__main__':
    {'prepare-model':prepare_model, 'science-reset-check':science_reset_check}[sys.argv[1]]()
