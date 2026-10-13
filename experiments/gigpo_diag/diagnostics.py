"""Read-only, process-sharded observations of the native training path.

Disabled unless GIGPO_DIAG_DIR is set. No environment calls or random draws.
Unknown state labels remain None; upstream rewards and return values are untouched.
"""
import functools
import gzip
import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

_update = 0
_batch_serial = 0
_adv_ids = []
_pending = {}
_optimizer_steps = 0


def enabled():
    return bool(os.environ.get("GIGPO_DIAG_DIR"))


def current_update():
    return _update


def safe(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if not enabled():
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            print(f"DIAGNOSTIC_UNAVAILABLE {fn.__name__}: {type(exc).__name__}: {exc}", flush=True)
            return None
    return wrapped


def plain(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if hasattr(x, "detach"):
        return x.detach().cpu().tolist()
    if hasattr(x, "tolist"):
        return x.tolist()
    if hasattr(x, "name") and hasattr(x, "arguments"):
        return {"predicate": str(x.name), "arguments": [str(a) for a in x.arguments]}
    return {"type": type(x).__name__, "repr": str(x)}


def digest(x):
    return hashlib.sha256(json.dumps(plain(x), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(kind, rows):
    root = Path(os.environ["GIGPO_DIAG_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{kind}.pid-{os.getpid()}.jsonl.gz"
    common = {"run_id": os.environ.get("GIGPO_RUN_ID"), "project": os.environ.get("GIGPO_ENV"),
              "code_commit": os.environ.get("GIGPO_CODE_COMMIT"), "policy_version": _update,
              "schema_version": "gigpo-diag-v1", "pid": os.getpid()}
    with gzip.open(destination, "at", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(plain({**common, **row}), ensure_ascii=False, allow_nan=False) + "\n")


def rate(values, scope):
    known = [bool(v) for v in values if v is not None]
    return {"numerator": sum(known), "denominator": len(known),
            "value": sum(known) / len(known) if known else None,
            "unknown_count": len(values) - len(known), "scope": scope}


@safe
def set_update(value):
    global _update
    _update = int(value)


def identity(info):
    return {"task_id": info.get("task_id", info.get("extra.gamefile", info.get("task_num"))),
            "variation_id": info.get("variation_id"), "reset_seed": info.get("reset_seed")}


@safe
def begin_rollout(obs, infos, uids, trajs, is_train, config):
    global _batch_serial
    _batch_serial += 1
    kind = "train_rollout" if is_train else "reference_eval"
    split = "train" if is_train else ("dev" if os.environ.get("GIGPO_ENV") == "appworld" else "reference_holdout")
    batch_id = f"u{_update}-{kind}-{_batch_serial}"
    groups = list(uids) if is_train else list(trajs)
    detail = is_train or split == "dev"
    all_detail = not is_train or _update + 1 in (1, 10, 30, 50, 100, int(config.trainer.total_epochs))
    selected = set(groups if all_detail else groups[:1])
    episodes, rows = [], []
    for i, info in enumerate(infos):
        row = {"rollout_batch_id": batch_id, "sample_kind": kind, "split": split,
               "episode_group_uid": str(groups[i]), "traj_uid": str(trajs[i]), **identity(info),
               "initial_anchor_hash": digest(obs["anchor"][i]),
               "native_score_initial": info.get("score"), "initial_state_hash": info.get("initial_state_hash"),
               "service_identity": info.get("service_identity"), "active_steps": 0,
               "reference_success": False, "strict_success": None, "env_reward_sum": 0.0,
               "format_valid_count": 0, "extraction_count": 0, "fallback_count": 0,
               "held_observed_objects": [],
               "events": {}, "termination": "budget", "detail_selected": detail and groups[i] in selected}
        episodes.append(row)
        rows.append({**row, **({"observation_initial": obs["anchor"][i], "reset_info": info} if row["detail_selected"] else {})})
    write("group_resets", rows)
    write("tasks_manifest", [{k: row[k] for k in ("rollout_batch_id", "sample_kind", "split", "episode_group_uid", "traj_uid", "task_id", "variation_id")} for row in rows])
    return {"episodes": episodes, "config": config, "prev_score": [i.get("score") for i in infos], "steps": [], "batch_id": batch_id}


def verify_groups(obs, infos, uids):
    """Training correctness guard, deliberately separate from optional logging."""
    grouped = defaultdict(list)
    for i, uid in enumerate(uids):
        grouped[str(uid)].append(digest({**identity(infos[i]), "anchor": obs["anchor"][i],
                                       "score": infos[i].get("score"), "state": infos[i].get("initial_state_hash")}))
    if any(len(set(hashes)) != 1 for hashes in grouped.values()):
        raise RuntimeError("Mixed task/variation/start in native collector uid; refusing advantage update")


def extraction(raw, env):
    tag = "code" if env == "appworld" else "action"
    text = raw.lower() if env == "alfworld" else raw
    start, end = text.find(f"<{tag}>"), text.find(f"</{tag}>")
    return start >= 0 and end >= start + len(tag) + 2


def events(command, feedback, info, env):
    out = {}
    low = feedback.lower()
    if env == "alfworld":
        for name, phrase in (("object_taken", "you pick up"), ("object_cleaned", "you clean"),
                             ("object_heated", "you heat"), ("object_cooled", "you cool"),
                             ("object_placed", "you put")):
            out[name] = phrase in low
        out["arrived_at_sink"] = command.startswith("go to sinkbasin") and "nothing happens" not in low and "you arrive" in low
        out["carrying_target_at_sink"] = None
    elif env == "scienceworld":
        out["focus_feedback"] = command.startswith("focus on ") and low.startswith("you focus")
        out["new_connection"] = command.startswith("connect ") and low.startswith("you connect") and "already" not in low
    else:
        out["cross_turn_result_used"] = None
        out["api_trace_ref"] = info.get("api_trace_ref")
    return out


@safe
def step(ctx, batch, batch_input, raw, obs, next_obs, rewards, dones, infos, active, step_index, tokenizer):
    if ctx is None:
        return
    import numpy as np
    env = os.environ.get("GIGPO_ENV")
    ids = [f"{ctx['batch_id']}:{e['traj_uid']}:{step_index}" for e in ctx["episodes"]]
    batch.non_tensor_batch["analysis_step_id"] = np.array(ids, dtype=object)
    batch.non_tensor_batch["diag_step_index"] = np.array([step_index] * len(ids), dtype=object)
    numeric, detailed = [], []
    for i, ep in enumerate(ctx["episodes"]):
        if not bool(active[i]):
            continue
        info = infos[i]
        feedback = str(next_obs["anchor"][i])
        command = str(info.get("diag_executed_action", ""))
        valid = bool(info["is_action_valid"]) if "is_action_valid" in info else None
        parsed = extraction(raw[i], env)
        score = info.get("score")
        done = bool(plain(dones[i]))
        won = bool(info.get("won", False))
        strict = info.get("strict_success") if env == "scienceworld" else (won if done or step_index + 1 == ctx["config"].env.max_steps else None)
        event = events(command, feedback, info, env)
        if env == "alfworld":
            held = ep["held_observed_objects"]
            taken = re.search(r"you pick up (?:the )?(.+?) from", feedback.lower())
            placed = re.search(r"you put (?:the )?(.+?) (?:in|on)", feedback.lower())
            if taken and taken[1] not in held:
                held.append(taken[1])
            if placed and placed[1] in held:
                held.remove(placed[1])
            target = re.search(r"in_recep-([^-]+)-", str(ep["task_id"]))
            if event["arrived_at_sink"] and target and any(x.startswith(target[1].lower()+" ") for x in held):
                event["carrying_target_at_sink"] = True
        low = feedback.lower()
        execution = "no_match" if any(x in low for x in ("nothing happens", "no known action", "i don't understand", "unknown action")) else "exception" if "traceback" in low else "unknown"
        row = {"analysis_step_id": ids[i], "rollout_batch_id": ctx["batch_id"], "sample_kind": ep["sample_kind"],
               "split": ep["split"], "episode_group_uid": ep["episode_group_uid"], "traj_uid": ep["traj_uid"],
               "task_id": ep["task_id"], "variation_id": ep["variation_id"], "step_index": step_index,
               "active_mask": True, "anchor_pre_hash": digest(obs["anchor"][i]),
               "generic_feedback_anchor": str(obs["anchor"][i]).strip().lower() in ("nothing happens.", "", "no known action matches that input."),
               "action_signature": digest(command), "command_short": command[:240],
               "feedback_short": feedback[:280], "event_evidence_source": "native_feedback",
               "event_evidence_ref": ids[i],
               "format_valid_native": valid, "extraction_success": parsed, "fallback": not parsed,
               "execution": execution, "done_raw": done, "reference_success": won, "strict_success": strict,
               "score_pre": ctx["prev_score"][i], "score_post": score,
               "env_reward": float(plain(rewards[i])), "events": event,
               "input_truncated": False if ctx["config"].data.truncation == "error" else None,
               "response_budget_reached": int(batch.batch["attention_mask"][i, -batch.batch["responses"].shape[-1]:].sum()) == ctx["config"].data.max_response_length,
               "label_version": "feedback-v1"}
        ep["active_steps"] += 1
        ep["env_reward_sum"] += row["env_reward"]
        ep["format_valid_count"] += int(valid is True)
        ep["extraction_count"] += int(parsed)
        ep["fallback_count"] += int(not parsed)
        ep["reference_success"] = ep["reference_success"] or won
        if strict is not None:
            ep["strict_success"] = bool(strict)
        ep["native_score_final"] = score
        for key, value in event.items():
            if isinstance(value, bool):
                ep["events"][key] = value or ep["events"].get(key, False)
            elif value is None:
                ep["events"].setdefault(key, None)
            else:
                ep["events"][key] = value
        if done:
            ep["termination"] = "task_success" if ep["strict_success"] else info.get("termination", "environment_done")
        ctx["prev_score"][i] = score
        ctx["steps"].append(row)
        numeric.append(row)
        if ep["detail_selected"]:
            inputs = batch_input.batch["input_ids"][i]
            input_mask = batch_input.batch["attention_mask"][i].bool()
            prompt_ids = inputs[input_mask].detach().cpu().tolist()
            response_ids = batch.batch["responses"][i].detach().cpu().tolist()
            detailed.append({**row, "rendered_prompt": tokenizer.decode(prompt_ids, skip_special_tokens=False),
                             "prompt_token_ids": prompt_ids, "response_token_ids": response_ids,
                             "raw_response": raw[i], "executed_command": command, "observation_post": feedback,
                             "native_info": info, "selection_reason": "preselected_group_slot_0_or_fixed_snapshot"})
    write("steps_numeric", numeric)
    if detailed:
        write("detailed_steps", detailed)


@safe
def end_rollout(ctx):
    if ctx is None:
        return
    write("episodes", ctx["episodes"])
    eps, steps = ctx["episodes"], ctx["steps"]
    metrics = {"reference_success": rate([e["reference_success"] for e in eps], "episode"),
               "strict_success": rate([e["strict_success"] for e in eps], "episode"),
               "format_valid": rate([s["format_valid_native"] for s in steps], "active_policy_step"),
               "extraction": rate([s["extraction_success"] for s in steps], "active_policy_step"),
               "fallback": rate([s["fallback"] for s in steps], "active_policy_step"),
               "generic_anchor": rate([s["generic_feedback_anchor"] for s in steps], "active_policy_step")}
    write("rollout_metrics", [{"rollout_batch_id": ctx["batch_id"], "sample_kind": eps[0]["sample_kind"],
                               "metrics": metrics, "episodes": len(eps), "active_steps": len(steps)}])


@safe
def reward_boundary(data, stage):
    global _adv_ids
    _adv_ids = list(data.non_tensor_batch.get("analysis_step_id", []))
    rows = []
    token = plain(data.batch["token_level_scores"].sum(-1))
    steps = plain(data.batch.get("step_rewards"))
    for i, value in enumerate(token):
        rows.append({"analysis_step_id": str(_adv_ids[i]) if _adv_ids else None,
                     "stage": stage, "token_score_sum": value, "step_return": steps[i] if steps is not None else None})
    write("reward_inputs", rows)


@safe
def set_advantage_ids(data):
    global _adv_ids
    _adv_ids = list(data.non_tensor_batch.get("analysis_step_id", []))


@safe
def capture_core(groups, step_groups, token_rewards, step_rewards, ae, astep, joint, mask, weight):
    global _pending
    m = mask.detach().bool()
    counts = m.sum(-1)
    def scalar(t):
        return plain((t.detach() * m).sum(-1) / counts.clamp(min=1))
    a, s, j = scalar(ae), scalar(astep), scalar(joint)
    err = plain(((joint.detach() - (ae.detach() + weight * astep.detach())).abs() * m).max())
    rows = [{"analysis_step_id": str(_adv_ids[i]) if _adv_ids else None,
             "training_row_index": i,
             "episode_group_uid": str(groups[i]), "step_group_id": str(step_groups[i]),
             "AE": a[i], "AS": s[i], "joint": j[i], "step_weight": weight,
             "response_mask_count": int(counts[i]), "token_reward_sum": float(token_rewards[i].detach().sum()),
             "actual_step_return": float(step_rewards[i].detach()), "reconstruction_error": err}
            for i in range(len(a))]
    write("advantages", rows)
    anchors = defaultdict(list)
    for i, row in enumerate(rows):
        anchors[row["step_group_id"]].append(row)
    write("anchor_summaries", [{"step_group_id": key, "episode_group_uid": values[0]["episode_group_uid"],
                                "step_count": len(values), "distinct_trajectories": len({x["analysis_step_id"].rsplit(":", 1)[0] for x in values if x["analysis_step_id"]}),
                                "return_min": min(x["actual_step_return"] for x in values),
                                "return_max": max(x["actual_step_return"] for x in values)} for key, values in anchors.items()])
    unique = list({r['analysis_step_id'] or str(i): r for i,r in enumerate(rows)}.values())
    _pending = {"diag/adv/joint_reconstruction_error": err,
                "diag/adv/joint_nonzero_step_rate": sum(abs(x['joint']) > 1e-6 for x in unique) / len(unique),
                "diag/adv/unique_active_steps": len(unique), "diag/adv/training_rows": len(rows),
                "diag/adv/AE_abs_mean": sum(abs(x['AE']) for x in unique) / len(unique),
                "diag/adv/AS_abs_mean": sum(abs(x['AS']) for x in unique) / len(unique)}


@safe
def update_completed(value, metrics):
    global _update
    _update = int(value)
    metrics.update(_pending)
    metrics["train/policy_update"] = _update
    write("updates", [{"policy_update": _update, "native_actor_update_returned": True}])
    if _update == 1 or _update % 5 == 0:
        import subprocess
        import sys
        root = os.environ['GIGPO_RUN_DIR']
        with open(Path(root)/'analysis.log', 'ab') as log:
            subprocess.run([sys.executable, '-m', 'experiments.gigpo_diag.analyze', root],
                           stdout=log, stderr=subprocess.STDOUT, timeout=45, check=False)


@safe
def parameter_sample(optimizer):
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.numel():
                t = p.detach().reshape(-1)
                stride = max(1, t.numel() // 256)
                return t[::stride][:256].float().cpu().clone()
    return None


@safe
def optimizer_completed(optimizer, before):
    global _optimizer_steps
    _optimizer_steps += 1
    after = parameter_sample(optimizer)
    write("optimizer", [{"optimizer_step_local": _optimizer_steps,
                          "parameter_sample_changed": bool((after != before).any()) if before is not None and after is not None else None,
                          "sample_max_abs_change": float((after - before).abs().max()) if before is not None and after is not None else None}])


@safe
def weights_loaded(source):
    write("weight_sync", [{"source": source, "optimizer_steps_completed_local": _optimizer_steps,
                            "event": "native_weight_load_returned", "timestamp": time.time()}])


@safe
def record_metrics(data, step):
    write("metrics", [{"global_step": int(step), "metrics": data}])


@safe
def checkpoint(path, step):
    write("checkpoints", [{"global_step": int(step), "path": str(path), "native_save_returned": True,
                           "sampler_resume": "ScienceWorld explicit RNG sidecar; other env state not bitwise replay guaranteed"}])


@safe
def tracker_status(state, **extra):
    write("tracker", [{"backend": "swanlab", "state": state, **extra}])


@safe
def save_sampler(trainer, checkpoint_dir):
    states = {}
    for key in ("envs", "val_envs"):
        inner = getattr(getattr(trainer, key), "envs", None)
        if hasattr(inner, "sampler_state"):
            states[key] = inner.sampler_state()
    if states:
        path = Path(checkpoint_dir) / "environment_sampler.json"
        path.write_text(json.dumps(plain(states)), encoding="utf-8")


@safe
def restore_sampler(trainer):
    if not trainer.global_steps:
        return
    checkpoint_dir = trainer.config.trainer.get("resume_from_path") or str(Path(trainer.config.trainer.default_local_dir) / f"global_step_{trainer.global_steps}")
    path = Path(checkpoint_dir) / "environment_sampler.json"
    if not path.exists():
        write("resume", [{"sampler_restored": False, "reason": "missing sidecar; environment replay not exact"}])
        return
    states = json.loads(path.read_text(encoding="utf-8"))
    for key, state in states.items():
        getattr(trainer, key).envs.restore_sampler_state(state)
    write("resume", [{"sampler_restored": True, "path": str(path)}])
