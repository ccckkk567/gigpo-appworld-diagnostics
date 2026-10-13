"""Synthetic correctness fixtures only; none are reported as HPC experiment data."""
import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_core_logging_is_read_only():
    diag=load(ROOT/'experiments/gigpo_diag/diagnostics.py','diag_fixture')
    package=types.ModuleType('experiments.gigpo_diag'); package.diagnostics=diag
    fake_verl=types.ModuleType('verl'); fake_verl.DataProto=object
    with mock.patch.dict(sys.modules,{'verl':fake_verl,'experiments.gigpo_diag':package}):
        core=load(ROOT/'gigpo/core_gigpo.py','core_fixture')
    kwargs={'token_level_rewards':torch.tensor([[0.,0.,10.],[0.,0.,-.1],[0.,0.,0.],[0.,0.,0.]]),
            'step_rewards':torch.tensor([10.,-.1,0.,0.]),'anchor_obs':np.array(['same','same','other','other'],dtype=object),
            'index':np.array(['g1','g1','g2','g2'],dtype=object),'mode':'mean_std_norm'}
    import inspect
    signature=inspect.signature(core.compute_gigpo_outcome_advantage)
    kwargs['response_mask' if 'response_mask' in signature.parameters else 'eos_mask']=torch.tensor([[1.,1.,1.]]*4)
    if 'traj_index' in signature.parameters: kwargs['traj_index']=np.array(['t1','t2','t3','t4'],dtype=object)
    with mock.patch.dict(os.environ,{},clear=False):
        os.environ.pop('GIGPO_DIAG_DIR',None)
        original=core.compute_gigpo_outcome_advantage(**kwargs)
    saved={k:v.clone() for k,v in kwargs.items() if isinstance(v,torch.Tensor)}
    with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ,{'GIGPO_DIAG_DIR':tmp}):
        state=torch.random.get_rng_state().clone()
        measured=core.compute_gigpo_outcome_advantage(**kwargs)
        assert torch.equal(state,torch.random.get_rng_state())
        for a,b in zip(original,measured): assert torch.equal(a,b)
        for k,v in saved.items(): assert torch.equal(v,kwargs[k]),k
        assert list(Path(tmp).glob('advantages.*.jsonl.gz'))
        assert diag.rate([None,True,False],'test')==dict(numerator=1,denominator=2,value=.5,unknown_count=1,scope='test')


def test_mixed_start_is_rejected():
    diag=load(ROOT/'experiments/gigpo_diag/diagnostics.py','diag_group_fixture')
    same=[dict(task_id='A',variation_id=1),dict(task_id='A',variation_id=1)]
    diag.verify_groups({'anchor':['same','same']},same,['g','g'])
    for infos,obs in [([same[0],dict(task_id='A',variation_id=2)],['same','same']), (same,['same','different'])]:
        try: diag.verify_groups({'anchor':obs},infos,['g','g'])
        except RuntimeError: pass
        else: raise AssertionError('Mixed start was not rejected')


def test_science_parent_payload_and_worker_mapping():
    path=ROOT/'agent_system/environments/env_package/sciworld/envs.py'
    if not path.exists(): return
    fake_mp=types.ModuleType('torch.multiprocessing')
    fake_gym=types.ModuleType('gymnasium'); fake_gym.Env=object
    with mock.patch.dict(sys.modules,{'gymnasium':fake_gym}):
        module=load(path,'science_fixture')
    class Pipe:
        def __init__(self): self.payload=None
        def send(self,message): self.payload=message[1]
        def recv(self):
            p=self.payload
            return 'same',dict(task_id=p['task_id'],variation_id=p['variation_id'],reset_seed=p['reset_seed'],available_actions=[],possible_actions=[],score=0)
    env=module.SciWorldMultiProcessEnv.__new__(module.SciWorldMultiProcessEnv)
    env.group_n=2; env.env_num=2; env.num_processes=4
    env.variations_idx=[[1,7],[2,5],[3,2]]; env._rng=np.random.RandomState(0); env._reset_count=0
    env._parent_remotes=[Pipe() for _ in range(4)]; env.prev_available_actions=[None]*4; env.prev_possible_actions=[None]*4; env._closed=True
    for repeat in range(2):
        obs,infos=env.reset()
        for a,b in ((0,1),(2,3)): assert env._parent_remotes[a].payload==env._parent_remotes[b].payload
    saved=env.sampler_state(); env.reset(); expected=[p.payload for p in env._parent_remotes]
    env.restore_sampler_state(saved); env.reset(); assert [p.payload for p in env._parent_remotes]==expected
    # Exercise the actual worker reset branch against a stand-in simulator.
    class Sim:
        loaded=[]
        def __init__(self,*a,**kw): pass
        def get_task_names(self): return ['zero','one','two','three']
        def load(self,*args): self.loaded.append(args)
        def reset(self): return 'state',{'score':0}
        def get_task_description(self): return 'task'
        def get_possible_actions(self): return []
        def get_possible_objects(self): return []
        def get_valid_action_object_combinations(self): return []
        def close(self): pass
    class WorkerPipe:
        def __init__(self): self.commands=iter([('reset',dict(task_id=2,variation_id=5,reset_seed=123)),('close',None)]); self.results=[]
        def recv(self): return next(self.commands)
        def send(self,x): self.results.append(x)
        def close(self): pass
    sci=types.ModuleType('scienceworld'); sci.ScienceWorldEnv=Sim
    p=WorkerPipe()
    with mock.patch.dict(sys.modules,{'scienceworld':sci}): module._worker(p,0,[],'easy',100,None,variations_idx=env.variations_idx)
    assert Sim.loaded==[('two',5,'easy')]
    assert p.results[0][1]['task_id']=='two' and p.results[0][1]['variation_id']==5


def test_app_service_count():
    from types import SimpleNamespace as N
    services=load(ROOT/'experiments/gigpo_diag/services.py','services_fixture')
    cfg=N(data=N(train_batch_size=16,val_batch_size=16),env=N(rollout=N(n=8)),actor_rollout_ref=N(rollout=N(val_kwargs=N(n=1))))
    assert services.required_instances(cfg)==144
    cfg.actor_rollout_ref.rollout.val_kwargs.n=2
    try: services.required_instances(cfg)
    except ValueError: pass
    else: raise AssertionError('inconsistent eval grouping accepted')


def test_step_capture_has_ids_raw_prompt_and_unknowns():
    import gzip
    from types import SimpleNamespace as N
    diag=load(ROOT/'experiments/gigpo_diag/diagnostics.py','diag_step_fixture')
    cfg=N(trainer=N(total_epochs=150),env=N(max_steps=2),data=N(truncation='error',max_response_length=3))
    class Tokenizer:
        def decode(self,ids,**kw): return 'rendered '+str(ids)
    batch=N(batch={'responses':torch.tensor([[3,4,5]]),'attention_mask':torch.ones((1,5),dtype=torch.long)},non_tensor_batch={})
    inputs=N(batch={'input_ids':torch.tensor([[1,2]]),'attention_mask':torch.ones((1,2),dtype=torch.long)})
    with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ,{'GIGPO_DIAG_DIR':tmp,'GIGPO_ENV':'alfworld','GIGPO_RUN_ID':'fixture'}):
        before={'anchor':['reset']}
        ctx=diag.begin_rollout(before,[{'extra.gamefile':'pick_clean_then_place_in_recep-Cloth-None-Toilet/trial'}],['g'],['t'],True,cfg)
        next_obs={'anchor':['You pick up the cloth 1 from the cabinet 1.']}
        diag.step(ctx,batch,inputs,['<think>x</think><action>take cloth 1 from cabinet 1</action>'],before,next_obs,
                  np.array([0.]),np.array([False]),[dict(diag_executed_action='take cloth 1 from cabinet 1',is_action_valid=True,won=False)],np.array([True]),0,Tokenizer())
        assert 'analysis_step_id' in batch.non_tensor_batch
        after={'anchor':['You arrive at sinkbasin 1.']}
        diag.step(ctx,batch,inputs,['<think>x</think><action>go to sinkbasin 1</action>'],next_obs,after,
                  np.array([0.]),np.array([False]),[dict(diag_executed_action='go to sinkbasin 1',is_action_valid=True,won=False)],np.array([True]),1,Tokenizer())
        diag.end_rollout(ctx)
        with gzip.open(next(Path(tmp).glob('detailed_steps.*')),'rt') as f: saved=[json.loads(line) for line in f]
        assert len(saved)==2 and saved[0]['raw_response'].startswith('<think>')
        assert saved[0]['rendered_prompt']=='rendered [1, 2]'
        assert saved[1]['events']['carrying_target_at_sink'] is True, (ctx['episodes'], saved[1]['events'])
        assert saved[0]['score_pre'] is None


if __name__=='__main__':
    tests=[test_native_core_logging_is_read_only,test_mixed_start_is_rejected,test_science_parent_payload_and_worker_mapping,test_app_service_count,test_step_capture_has_ids_raw_prompt_and_unknowns]
    for test in tests:
        test(); print('PASS',test.__name__,flush=True)
