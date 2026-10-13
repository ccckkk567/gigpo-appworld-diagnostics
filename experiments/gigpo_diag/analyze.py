"""Incremental local report from actual diagnostic records; no synthetic curves."""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path


def rows(root, name):
    for path in sorted((root/'diagnostics').glob(name+'.pid-*.jsonl.gz')):
        try:
            with gzip.open(path,'rt',encoding='utf-8') as f:
                for line in f:
                    try: yield json.loads(line)
                    except json.JSONDecodeError: continue
        except (EOFError,OSError):
            pass  # active writer may not have closed the current gzip member yet


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--exit-code',type=int)
    args=parser.parse_args()
    root=args.run_dir
    tables=root/'tables'; tables.mkdir(exist_ok=True)
    eps=list(rows(root,'episodes'))
    updates=list(rows(root,'updates'))
    metrics=list(rows(root,'rollout_metrics'))
    max_update=max((x['policy_update'] for x in updates),default=0)
    updates_this_run=len({x['policy_update'] for x in updates})
    path=root/'RUN_MANIFEST.json'
    manifest=json.loads(path.read_text()) if path.exists() else {}
    status='partial' if max_update else 'not_yet_updated'
    if args.exit_code is not None:
        status='completed' if args.exit_code==0 and max_update else ('failed_after_updates' if max_update else 'failed_before_updates')
    manifest.update(status=status,policy_updated=updates_this_run>0,training_updates=updates_this_run,policy_version=max_update,exit_code=args.exit_code)
    path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    flat=[]
    for batch in metrics:
        for name,m in batch['metrics'].items():
            flat.append({'policy_version':batch['policy_version'],'sample_kind':batch['sample_kind'],
                         'rollout_batch_id':batch['rollout_batch_id'],'metric':name,**m})
    grouped=defaultdict(list)
    for e in eps:
        grouped[(e['policy_version'],e['sample_kind'],e['rollout_batch_id'])].append(e)
    for (version,kind,batch),episodes in grouped.items():
        names=set().union(*(set(e.get('events',{})) for e in episodes))
        for name in names:
            values=[e.get('events',{}).get(name) for e in episodes]
            known=[v for v in values if isinstance(v,bool)]
            if any(v is not None and not isinstance(v,bool) for v in values): continue
            flat.append(dict(policy_version=version,sample_kind=kind,rollout_batch_id=batch,metric='phase/'+name,
                             numerator=sum(known),denominator=len(known),value=sum(known)/len(known) if known else None,
                             unknown_count=len(values)-len(known),scope='episode'))
    with (tables/'rollout_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['policy_version','sample_kind','rollout_batch_id','metric','numerator','denominator','value','unknown_count','scope']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(flat)
    figures=[]
    if flat:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        (root/'figures').mkdir(exist_ok=True)
        phase_keys=sorted({r['metric'] for r in flat if r['metric'].startswith('phase/')})
        for name,keys in [('learning_outcomes',['reference_success','strict_success']),('protocol_to_execution',['format_valid','extraction','fallback']),('anchor_evolution',['generic_anchor']),('phase_learning',phase_keys)]:
            fig,ax=plt.subplots(figsize=(8,4.5))
            for key in keys:
                for kind in ('train_rollout','reference_eval'):
                    points=[r for r in flat if r['metric']==key and r['sample_kind']==kind and r['value'] is not None]
                    points.sort(key=lambda r:r['policy_version'])
                    if points: ax.plot([p['policy_version'] for p in points],[p['value'] for p in points],marker='o',label=f'{kind}: {key}')
            ax.set(xlabel='Policy updates completed before rollout',ylabel='Rate among known observations',ylim=(-.03,1.03),title=f"{manifest.get('env','unknown')} / GiGPO / Qwen2.5-1.5B / seed 0\n{len(eps)} episodes; denominators and unknowns in CSV")
            if ax.lines: ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(root/'figures'/f'{name}.png',dpi=170); plt.close(fig)
            figures.append(name)
    report=root/'reports'; report.mkdir(exist_ok=True)
    cases=[e for e in eps if e.get('detail_selected')][:3]
    body=[f"# {manifest.get('env','unknown')} / {root.name}",f"Status: **{status}**. Native actor updates returned in this run: **{updates_this_run}**; latest absolute policy version: **{max_update}**.",
          f"Recorded episodes: {len(eps)}. No model update or trajectory is inferred from a submitted job.",
          "## Evidence", "- Full counts, denominator, unknown count and sample kind: `tables/rollout_metrics.csv`.",
          "- Actual AE/AS/joint, native reward inputs before/after penalty: `diagnostics/advantages.*` and `reward_inputs.*`.",
          "- Parameter samples and actual weight-load return path: `diagnostics/optimizer.*` and `weight_sync.*`.",
          "- Full preselected groups: `diagnostics/detailed_steps.*`; reference holdout long text is excluded.",
          "- Native checkpoints: `checkpoints/`; metadata index: `diagnostics/checkpoints.*`.",
          "## Cases (preselected, not outcome-selected)"]
    body.extend(f"- {e['traj_uid']} / {e['task_id']}: reference={e['reference_success']}, strict={e['strict_success']}, steps={e['active_steps']}." for e in cases)
    body.extend(["## Current limits", "Strict/phase unknowns are not failures. No causal or cross-project performance claim is made. Fixed study slices and prefix probes remain pending; they do not block native learning.",
                 "## Next review", "Verify the first optimizer changes and next native weight synchronization, then inspect protocol and actual object/connection/API behavior by policy version. Continue the reference schedule; do not launch extra algorithms automatically."])
    (report/'current.md').write_text('\n\n'.join(body)+'\n',encoding='utf-8')
    if max_update:
        (report/f'update_{max_update:03d}.md').write_text('\n\n'.join(body)+'\n',encoding='utf-8')
    print(json.dumps({'status':status,'updates':max_update,'episodes':len(eps),'figures':figures,'report':str(report/'current.md')}))


if __name__=='__main__': main()
