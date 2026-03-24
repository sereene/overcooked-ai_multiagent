import os
import argparse
import torch
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import DQNConfig 
from ray.air.integrations.wandb import WandbLoggerCallback
from datetime import datetime

from env_wrapper import Rllib_multi_agent
from callbacks import VideoCallbacks

def env_creator(env_config):
    env = Rllib_multi_agent(env_config)
    return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fc_size", type=int, default=64, help="FC Layer size")
    args = parser.parse_args()

    ray.init()

    # 환경 등록
    env_name = "overcooked_dqn_shared_reward_shaping"
    register_env(env_name, env_creator)

    dummy_env = Rllib_multi_agent({"layout_name": "asymmetric_advantages"})
    
    single_obs_space = dummy_env.observation_space["agent_0"]
    single_act_space = dummy_env.action_space["agent_0"]
    
    dummy_env.close()

    current_dir = os.getcwd()
    local_log_dir = os.path.join(current_dir, "results")
    experiment_name = f"DQN_Overcooked_Shared_asymmetric_advantages_reward_shaping_1e-5"

    start_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    gif_save_path = os.path.join(local_log_dir, experiment_name, f"videos_{start_time}")

    config = (
        DQNConfig()  
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
        .environment(
            env=env_name,
            env_config={
                "layout_name": "asymmetric_advantages", 
                "reward_shaping": True,
                "horizon": 500  
            },
            disable_env_checking=True
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=8,
        )
        .multi_agent(
            policies={
                "shared_policy": (None, single_obs_space, single_act_space, {})
            },
            policy_mapping_fn=lambda agent_id, *ep_args, **kwargs: "shared_policy"
        )
        .training(
            gamma=0.99,
            lr=1e-5, 
            train_batch_size=256,
            target_network_update_freq=500, # 타겟 네트워크 업데이트 주기
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": 50000, # 리플레이 버퍼 크기 설정
            },
            # model={
            #     "fcnet_hiddens": [args.fc_size, args.fc_size],
            #     # use_lstm 등의 설정은 아예 삭제하거나 기본값(False)으로 유지
            # },
        )
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.01,       
                "epsilon_timesteps": 15_000_000, 
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

    tune.run(
        "DQN", 
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
                group="dqn_experiment",
                job_type="training",
                name=experiment_name,
                log_config=True
            )
        ]
    )