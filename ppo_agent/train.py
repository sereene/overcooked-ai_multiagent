import os
import argparse
import torch
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig  # QMIX -> PPO 변경
from ray.air.integrations.wandb import WandbLoggerCallback
from datetime import datetime
from td_surprise_callbacks import TDSpikeVideoCallback

# 작성한 환경과 콜백 임포트
from env_wrapper import Rllib_multi_agent
from callbacks import VideoCallbacks

# ==========================================
# 1. 환경 생성 함수 (PPO용)
# ==========================================
def env_creator(env_config):
    # PPO는 QMIX처럼 에이전트를 Tuple/Group으로 묶을 필요가 없습니다.
    env = Rllib_multi_agent(env_config)
    return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fc_size", type=int, default=64, help="FC Layer/LSTM size")
    args = parser.parse_args()

    # Ray 초기화
    ray.init()

    # 환경 등록
    env_name = "overcooked_ppo_shared_reward_shaping_5e-6_fc64"
    register_env(env_name, env_creator)

    # 더미 환경을 통해 단일 에이전트의 행동/관측 공간 추출
    dummy_env = Rllib_multi_agent({"layout_name": "asymmetric_advantages"})
    
    single_obs_space = dummy_env.observation_space["agent_0"]
    single_act_space = dummy_env.action_space["agent_0"]
    
    dummy_env.close()

    # ==========================================
    # 2. 로깅 디렉토리 및 시간 포맷 설정
    # ==========================================
    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results")
    experiment_name = f"PPO_Overcooked_Shared_asymmetric_advantages_reward_shaping_5e-6"

    start_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    gif_save_path = os.path.join(local_log_dir, experiment_name, f"videos_{start_time}")

    config = (
        PPOConfig()  
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
        .environment(
            env=env_name,
            env_config={
                "layout_name": "asymmetric_advantages", 
                "reward_shaping": True,
                "horizon": 500  # <--- 이 줄을 추가하세요!
            },
            disable_env_checking=True
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=8, 
            # PPO는 On-policy이므로 fragment 길이를 QMIX보다 길게 가져가는 것이 안정적입니다.
            rollout_fragment_length=256,
        )
        .multi_agent(
            policies={
                # Shared Policy 설정: 하나의 정책 공간만 정의
                "shared_policy": (None, single_obs_space, single_act_space, {})
            },
            # 모든 에이전트("agent_0", "agent_1")를 "shared_policy"로 매핑
            policy_mapping_fn=lambda agent_id, *ep_args, **kwargs: "shared_policy"
        )
        .training(
            gamma=0.99,
            lr=5e-6,
            train_batch_size=8*256,  # num_workers * rollout_fragment_length
            sgd_minibatch_size=128,
            num_sgd_iter=10,
            # model={
            #     "fcnet_hiddens": [args.fc_size, args.fc_size],
            #     "use_lstm": True,      
            #     "max_seq_len": 20,     # BPTT 길이 설정
            #     "lstm_cell_size": args.fc_size,
            # },
        )
        .evaluation(
            evaluation_interval=10, 
            evaluation_duration=20, 
            evaluation_duration_unit="episodes",
            evaluation_config={
                "explore": False, 
            },
        )
        .callbacks(lambda: TDSpikeVideoCallback(
            out_dir=gif_save_path, 
            env_creator_fn=env_creator,
            every_n_evals=5,
            td_threshold=3.0,  # 이 값을 조절하여 '얼마나 튀었을 때' 저장할지 결정합니다.
            n_steps=15         # 에러가 발생한 시점 앞뒤 15스텝(총 30스텝)을 녹화합니다.
        ))
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
    )

    print(f"### Training Logs will be saved at: {local_log_dir} ###")
    print(f"### Videos will be saved at: {gif_save_path} ###")

    # ==========================================
    # 3. Ray Tune 실행
    # ==========================================
    tune.run(
        "PPO",  # "QMIX" -> "PPO" 변경
        name=experiment_name,
        stop={"timesteps_total": 30_000_000},
        local_dir=local_log_dir,
        metric="episode_reward_mean",
        mode="max",
        checkpoint_freq=500,
        checkpoint_at_end=True,
        keep_checkpoints_num=2,
        checkpoint_score_attr="episode_reward_mean",
        
        config=config.to_dict(),
        callbacks=[
            WandbLoggerCallback(
                project="overcooked_project", 
                group="ppo_experiment",
                job_type="training",
                name=experiment_name,
                log_config=True
            )
        ]
    )