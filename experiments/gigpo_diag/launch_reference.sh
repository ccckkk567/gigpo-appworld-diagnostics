#!/usr/bin/env bash
set -euo pipefail
ENGINE=${1:-vllm}
if (( $# )); then shift; fi
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

train_data_size=16
val_data_size=16
group_size=8
mode="mean_norm" # "mean_norm" or "mean_std_norm"
N_GPUS=${GIGPO_NUM_GPUS:-2}
if [[ ! -f "$GIGPO_DATA_DIR/text/train.parquet" || ! -f "$GIGPO_DATA_DIR/text/test.parquet" ]]; then
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size \
    --local_dir "$GIGPO_DATA_DIR"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$GIGPO_DATA_DIR/text/train.parquet \
    data.val_files=$GIGPO_DATA_DIR/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=7167 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$GIGPO_MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    env.env_name=AppWorld \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    trainer.critic_warmup=0 \
    "trainer.logger=[console,swanlab]" \
    trainer.project_name=gigpo-training-diagnostics \
    trainer.experiment_name="$GIGPO_RUN_ID" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    trainer.default_local_dir="$GIGPO_RUN_DIR/checkpoints" \
    trainer.resume_mode=${GIGPO_RESUME_MODE:-disable} \
    trainer.resume_from_path=${GIGPO_RESUME_FROM:-null} \
    algorithm.filter_groups.enable=False \
    ray_init.num_cpus=${GIGPO_CPUS:-32} \
    algorithm.gigpo.enable_similarity=False \
    env.multi_episode_rollout.enable=False \
    env.success_coef=0 \
    env.prompt_type=vanilla \
    +env.appworld.eval_split=dev \
    "$@"