import os
import argparse
import torch
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.a2c import A2CConfig  # 🔥 PPO -> A2C 변경
from ray.air.integrations.wandb import WandbLoggerCallback
from datetime import datetime

# 작성한 환경과 콜백 임포트
from env_wrapper import Rllib_multi_agent
from callbacks import VideoCallbacks

# ==========================================
# 1. 환경 생성 함수
# ==========================================
def env_creator(env_config):
    env = Rllib_multi_agent(env_config)
    return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fc_size", type=int, default=64, help="FC Layer/LSTM size")
    args = parser.parse_args()

    # Ray 초기화
    ray.init()

    # 환경 등록
    env_name = "overcooked_a2c_shared_reward_shaping_1e-5_fc64"
    register_env(env_name, env_creator)

    # 더미 환경을 통해 단일 에이전트의 행동/관측 공간 추출
    dummy_env = Rllib_multi_agent({"layout_name": "cramped_room"})
    
    single_obs_space = dummy_env.observation_space["agent_0"]
    single_act_space = dummy_env.action_space["agent_0"]
    
    dummy_env.close()

    # ==========================================
    # 2. 로깅 디렉토리 및 시간 포맷 설정
    # ==========================================
    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results")
    experiment_name = f"A2C_Overcooked_Shared_cramped_room_reward_shaping_1e-5" # 🔥 PPO -> A2C 이름 변경

    start_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    gif_save_path = os.path.join(local_log_dir, experiment_name, f"videos_{start_time}")

    # ==========================================
    # 3. A2C Config 설정 (MAA2C)
    # ==========================================
    config = (
        A2CConfig()  # 🔥 PPOConfig() -> A2CConfig() 변경
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
        .environment(
            env=env_name,
            env_config={
                "layout_name": "cramped_room", 
                "reward_shaping": True
            },
            disable_env_checking=True
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=8, 
            rollout_fragment_length=256,
        )
        .multi_agent(
            policies={
                # Shared Policy 설정 (이 설정 덕분에 MAA2C처럼 하나의 거대한 Critic/Actor를 공유하게 됨)
                "shared_policy": (None, single_obs_space, single_act_space, {})
            },
            policy_mapping_fn=lambda agent_id, *ep_args, **kwargs: "shared_policy"
        )
        .training(
            gamma=0.99,
            lr=1e-4,
            train_batch_size=8*256,  # num_workers * rollout_fragment_length
            
            # 🔥 PPO 전용 옵션(num_sgd_iter, sgd_minibatch_size) 삭제
            # 🔥 A2C 필수 옵션 추가
            use_gae=True,            # Advantage(GAE) 사용 명시
            entropy_coeff=0.01,      # 에이전트가 다양한 행동을 시도하도록 유도 (탐험 보너스)
            
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
        .callbacks(lambda: VideoCallbacks(
            out_dir=gif_save_path, 
            env_creator_fn=env_creator,
            every_n_evals=5 
        ))
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
    )

    print(f"### Training Logs will be saved at: {local_log_dir} ###")
    print(f"### Videos will be saved at: {gif_save_path} ###")

    # ==========================================
    # 4. Ray Tune 실행
    # ==========================================
    tune.run(
        "A2C",  # 🔥 "PPO" -> "A2C" 변경
        name=experiment_name,
        stop={"timesteps_total": 100000_000_000},
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
                group="a2c_experiment", # 🔥 wandb 그룹명 변경
                job_type="training",
                name=experiment_name,
                log_config=True
            )
        ]
    )