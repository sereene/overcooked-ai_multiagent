import os
import argparse
from model import CustomQMIXModel
import torch
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.qmix import QMixConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from gymnasium.spaces import Tuple
from datetime import datetime
from ray.rllib.models import ModelCatalog


# 작성한 환경과 콜백 임포트
from env_wrapper import Rllib_multi_agent
from callbacks import VideoCallbacks

# ==========================================
# 1. 환경 생성 함수 (QMIX용 에이전트 묶기)
# ==========================================
# ==========================================

# 1. 환경 생성 함수 (QMIX용 에이전트 묶기)
# ==========================================
def env_creator(env_config):
    env = Rllib_multi_agent(env_config)
    
    # QMIX를 위한 그룹화
    grouping = {
        "group_1": ["agent_0", "agent_1"],
    }
    
    # 개별 에이전트의 공간(Space)만 추출
    single_obs_space = env.observation_space["agent_0"]
    single_act_space = env.action_space["agent_0"]
    
    # 👇 [주의] 이 두 줄이 누락되어서 발생한 에러입니다! 반드시 포함되어야 합니다.
    obs_space = Tuple([single_obs_space, single_obs_space])
    act_space = Tuple([single_act_space, single_act_space])

    return env.with_agent_groups(grouping, obs_space=obs_space, act_space=act_space)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fc_size", type=int, default=64, help="FC Layer/LSTM size")
    args = parser.parse_args()

    # Ray 초기화
    ray.init()

    ModelCatalog.register_custom_model("custom_gru", CustomQMIXModel)

    # 환경 등록
    env_name = "overcooked_qmix_reward_shaping_1e-5_fc64"
    register_env(env_name, env_creator)

    # 더미 환경을 통해 그룹의 행동/관측 공간 추출
    # 수정됨: Box 객체 에러 해결을 위해 단일 공간을 Tuple로 직접 묶음
    # 더미 환경을 통해 그룹의 행동/관측 공간 추출
    dummy_env = Rllib_multi_agent({"layout_name": "asymmetric_advantages"})
    
    single_obs_space = dummy_env.observation_space["agent_0"]
    single_act_space = dummy_env.action_space["agent_0"]
    
    # 👇 [추가] 정책(Policy)에 전달할 그룹(Tuple) 공간을 명시적으로 만듭니다.
    from gymnasium.spaces import Tuple # (상단에 import 되어 있다면 생략 가능)
    group_obs_space = Tuple([single_obs_space, single_obs_space])
    group_act_space = Tuple([single_act_space, single_act_space])
    
    dummy_env.close()
    # ==========================================
    # 2. 로깅 디렉토리 및 시간 포맷 설정
    # ==========================================
    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results")
    experiment_name = f"QMIX_Overcooked_Qmix_asymmetric_advantages_reward_shaping_1e-5"

    start_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    gif_save_path = os.path.join(local_log_dir, experiment_name, f"videos_{start_time}")

    config = (
        QMixConfig()
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
        .environment(
            env=env_name,
            env_config={
                "layout_name": "asymmetric_advantages", 
                "reward_shaping": True
            },
            disable_env_checking=True
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=4, 
            rollout_fragment_length=20,
        )
        .multi_agent(
            policies={
                "group_1": (None, group_obs_space, group_act_space, {})
            },
            policy_mapping_fn=lambda agent_id, *ep_args, **kwargs: "group_1"
        )
        .training(
            mixer="vdn", 
            double_q=True, 
            grad_clip=20.0, 
            # model={
            #     "custom_model": "custom_gru", 
            #     "custom_model_config": {
            #         "fc_size": args.fc_size 
            #     },
            #     "max_seq_len": 100, 
            # },
            train_batch_size=64, 
            target_network_update_freq=500,
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": 50000, 
            },
            gamma=0.99,
            lr=1e-5,
        )
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.01,       
                "epsilon_timesteps": 10_000_000, 
            }
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
    # 3. Ray Tune 실행
    # ==========================================
    tune.run(
        "QMIX",
        name=experiment_name,
        stop={"timesteps_total": 20_000_000},
        local_dir=local_log_dir,
        # 👇 여기 두 줄을 "episode_reward_mean"으로 변경합니다.
        metric="episode_reward_mean",
        mode="max",
        checkpoint_freq=500,
        checkpoint_at_end=True,
        keep_checkpoints_num=2,
        checkpoint_score_attr="episode_reward_mean", # 👇 여기도 변경!
        
        config=config.to_dict(),
        callbacks=[
            WandbLoggerCallback(
                project="overcooked_project", 
                group="qmix_experiment",
                job_type="training",
                name=experiment_name,
                log_config=True
            )
        ]
    )