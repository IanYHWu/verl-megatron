NNODES=1
NGPUS_PER_NODES=8

echo "Using $NNODES nodes for training..."

project_name='reasoning_cache'
exp_name='vanilla_grpo_16k_inst_hard_30b_sync'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.285
max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 16))

loss_agg_mode="token-mean"

train_prompt_bsz=64
n_resp_per_prompt=8
train_prompt_mini_bsz=32
train_ppo_micro_batch_size_per_gpu=1
infer_ppo_micro_batch_size_per_gpu=2

MODEL_PATH=Qwen/Qwen3-30B-A3B-Instruct-2507

RAY_DATA_HOME="/home/ianwu/data"
TRAIN_FILE=$RAY_DATA_HOME/acemath_rl_30b_a3b_inst_hard_train.parquet
TEST_FILE=$RAY_DATA_HOME/acemath_rl_30b_a3b_inst_hard_test.parquet

temperature=1.0
top_p=1.0
top_k=-1
val_top_p=1.0

use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
offload=True

optimizer_offload_fraction=${OFFLOAD_FRACTION:-1.}

COMMON_PP=${COMMON_PP:-1}
COMMON_VPP=${COMMON_VPP:-null}
COMMON_CP=${COMMON_CP:-1}
COMMON_TP=${COMMON_TP:-4}
COMMON_EP=${COMMON_EP:-16}
COMMON_ETP=${COMMON_ETP:-1}

TRAIN_TP=${TRAIN_TP:-$COMMON_TP}
INFER_TP=${INFER_TP:-4}

ACTOR_PP=${ACTOR_PP:-$COMMON_PP}
ACTOR_VPP=${ACTOR_VPP:-$COMMON_VPP}
ACTOR_CP=${ACTOR_CP:-$COMMON_CP}
ACTOR_TP=${ACTOR_TP:-$TRAIN_TP}
ACTOR_EP=${ACTOR_EP:-$COMMON_EP}
ACTOR_ETP=${ACTOR_ETP:-$COMMON_ETP}
ROLLOUT_TP=${ROLLOUT_TP:-$INFER_TP}
REF_PP=${REF_PP:-$COMMON_PP}
REF_VPP=${REF_VPP:-$COMMON_VPP}
REF_CP=${REF_CP:-$COMMON_CP}
REF_TP=${REF_TP:-$TRAIN_TP}
REF_EP=${REF_EP:-$COMMON_EP}
REF_ETP=${REF_ETP:-$COMMON_ETP}
CRITIC_PP=${CRITIC_PP:-$COMMON_PP}
CRITIC_VPP=${CRITIC_VPP:-$COMMON_VPP}
CRITIC_CP=${CRITIC_CP:-$COMMON_CP}
CRITIC_TP=${CRITIC_TP:-$TRAIN_TP}
CRITIC_EP=${CRITIC_EP:-$COMMON_EP}
CRITIC_ETP=${CRITIC_ETP:-$COMMON_ETP}
RM_PP=${RM_PP:-$COMMON_PP}
RM_VPP=${RM_VPP:-$COMMON_VPP}
RM_CP=${RM_CP:-$COMMON_CP}
RM_TP=${RM_TP:-$TRAIN_TP}
RM_EP=${RM_EP:-$COMMON_EP}
RM_ETP=${RM_ETP:-$COMMON_ETP}

USE_MBRIDGE=True
USE_DIST_CKPT=False



python3 -m projects.reasoning_cache.main_ppo_reasoning_cache --config-name='ppo_megatron_trainer_rc' \
    data.train_files=\"${TRAIN_FILE}\" \
    data.val_files=\"${TEST_FILE}\" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path=\"${MODEL_PATH}\" \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.megatron.use_mbridge=${USE_MBRIDGE} \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP} \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${ACTOR_VPP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ACTOR_ETP} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type='alltoall' \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${infer_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${INFER_TP} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${infer_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${REF_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${REF_PP} \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${REF_VPP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${REF_CP} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${REF_EP} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${REF_ETP} \
    reward_model.reward_manager=naive \
    trainer.logger=['console','wandb'] \
    trainer.project_name=\"${project_name}\" \
    trainer.experiment_name=\"${exp_name}\" \
    trainer.n_gpus_per_node=\"${NGPUS_PER_NODES}\" \
    trainer.nnodes=\"${NNODES}\" \
    trainer.val_before_train=False \
    trainer.test_freq=50 \
    trainer.save_freq=10 \
    trainer.total_epochs=10 \
    trainer.resume_mode=disable \
    trainer.log_val_generations=2 \
    custom_reward_function.path=/home/ianwu/code/verl-megatron/verl/utils/reward_score/math_general.py \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.default_local_dir=/tmp/ianwu/checkpoints/${project_name}/${exp_name} \
    reasoning_cache.max_thinking_tokens=2048 \
    reasoning_cache.max_summary_tokens=2048 \
    reasoning_cache.omit_online_rollout_initial_thinking_step=False \
    reasoning_cache.use_think_tags=False \
    reasoning_cache.online_rollout_steps=2 \
    reasoning_cache.online_rollout_n_samples=1 \
    reasoning_cache.online_rollout_steps_val=2 \
    reasoning_cache.online_rollout_n_samples_val=4 \
    reasoning_cache.thinking_train_samples_per_online_rollout=2 \
    reasoning_cache.summary_train_samples_per_online_rollout=0 \
    reasoning_cache.thinking_train_n_samples=2 \
    reasoning_cache.summary_train_n_samples=2 \
    reasoning_cache.thinking_reward_rollout_steps=0 \
    reasoning_cache.summary_reward_rollout_steps=0 \
    reasoning_cache.thinking_reward_n_samples=1 \
    reasoning_cache.summary_reward_n_samples=4 \
    reasoning_cache.use_train_state_counter=True \
    reasoning_cache.use_replay_buffer=False \
    reasoning_cache.buffer_sample_summaries_per_problem=1 \
    reasoning_cache.buffer_purge_existing_on_add=True \
    reasoning_cache.save_rollouts=False