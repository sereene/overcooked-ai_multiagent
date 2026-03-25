import os
import numpy as np
import torch
import imageio.v2 as imageio
import pygame
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm

from env_wrapper import Rllib_multi_agent

def calculate_td_error(reward, v_t, v_next, is_done, gamma=0.99):
    if is_done:
        return abs(reward - v_t)
    return abs(reward + gamma * v_next - v_t)

def get_value_prediction(policy, obs, state_in):
    _, _, info = policy.compute_single_action(obs, state=state_in, explore=False)
    
    if "vf_preds" in info:
        return info["vf_preds"]
    else:
        input_dict = {"obs": torch.tensor([obs], dtype=torch.float32).to(policy.device)}
        state_t = [torch.tensor([s], dtype=torch.float32).to(policy.device) for s in state_in] if state_in else []
        with torch.no_grad():
            policy.model(input_dict, state_t, torch.tensor([1]))
            return policy.model.value_function()[0].item()

def evaluate_and_record_td_spikes(
    checkpoint_path: str, 
    layout_name: str = "asymmetric_advantages", 
    num_episodes: int = 10, 
    n_steps: int = 15, 
    z_threshold: float = 3.0, 
    min_td_threshold: float = 1.0
):
    
    ray.init(ignore_reinit_error=True)
    register_env("overcooked_ppo_shared_reward_shaping_5e-6_fc64", lambda config: Rllib_multi_agent(config))
    
    print(f"Loading checkpoint from: {checkpoint_path}...")
    algo = Algorithm.from_checkpoint(checkpoint_path)
    policy = algo.get_policy("shared_policy")
    
    env = Rllib_multi_agent({"layout_name": layout_name, "horizon": 500})
    
    out_dir = os.path.join(os.getcwd(), "td_spike_videos_eval")
    os.makedirs(out_dir, exist_ok=True)

    # 2. 에피소드 루프
    for ep in range(num_episodes):
        obs, _ = env.reset()
        
        # RNN 초기 상태 세팅
        rnn_states = {}
        if hasattr(policy, "get_initial_state"):
            rnn_states["agent_0"] = policy.get_initial_state()
            rnn_states["agent_1"] = policy.get_initial_state()

        states_buffer = []  # 렌더링용 State 객체 저장
        td_errors = []      # 에피소드 내 TD Error 기록
        
        # 첫 스텝의 State 캡처
        states_buffer.append(env.overcooked_env.state.deepcopy())
        step_i = 0
        
        while True:
            actions = {}
            v_preds = {}

            for agent_id, agent_obs in obs.items():
                state_in = rnn_states.get(agent_id, [])
                pass_state = state_in if len(state_in) > 0 else None
                
                res = algo.compute_single_action(
                    agent_obs, state=pass_state, policy_id="shared_policy", explore=False
                )
                if isinstance(res, tuple):
                    action, state_out, _ = res
                else:
                    action, state_out = res, []
                
                actions[agent_id] = action
                rnn_states[agent_id] = state_out
                
                # Value 추출
                v_preds[agent_id] = get_value_prediction(policy, agent_obs, pass_state)

            next_obs, rewards, done_dict, trunc_dict, _ = env.step(actions)
            is_done = done_dict.get("__all__", False) or trunc_dict.get("__all__", False)
            
            # 다음 상태의 가치(Value) 예측
            v_next_preds = {"agent_0": 0.0, "agent_1": 0.0}
            if not is_done:
                for agent_id, agent_obs in next_obs.items():
                    state_in = rnn_states.get(agent_id, [])
                    pass_state = state_in if len(state_in) > 0 else None
                    v_next_preds[agent_id] = get_value_prediction(policy, agent_obs, pass_state)
            
            td_err = calculate_td_error(rewards["agent_0"], v_preds["agent_0"], v_next_preds["agent_0"], is_done)
            td_errors.append(td_err)
            
            obs = next_obs
            states_buffer.append(env.overcooked_env.state.deepcopy())
            step_i += 1
            
            if is_done or step_i >= 500:
                break

        td_array = np.array(td_errors)
        mean_td = np.mean(td_array)
        std_td = np.std(td_array)
        
        spike_indices = np.where((td_array > mean_td + z_threshold * std_td) & (td_array > min_td_threshold))[0]
        
        if len(spike_indices) > 0:
            # 해당 에피소드에서 가장 TD 에러가 큰 지점을 선택
            max_spike_idx = spike_indices[np.argmax(td_array[spike_indices])]
            max_td_val = td_array[max_spike_idx]
            
            print(f"[Ep {ep+1}/{num_episodes}] TD Spike Detected at step {max_spike_idx} "
                  f"(TD: {max_td_val:.2f} | Mean: {mean_td:.2f} | Std: {std_td:.2f})")
            
            # 동영상 렌더링 및 저장
            start_t = max(0, max_spike_idx - n_steps)
            end_t = min(len(states_buffer) - 1, max_spike_idx + n_steps)
            
            frames = []
            for t in range(start_t, end_t + 1):
                state_obj = states_buffer[t]
                surface = env.visualizer.render_state(
                    state=state_obj,
                    grid=env.overcooked_env.mdp.terrain_mtx
                )
                frame = pygame.surfarray.pixels3d(surface)
                frame = np.transpose(frame, (1, 0, 2))
                
                h, w, c = frame.shape
                if h % 2 != 0: frame = frame[:-1, :, :]
                if w % 2 != 0: frame = frame[:, :-1, :]
                
                frames.append(frame.astype(np.uint8))
                
            if frames:
                filename = f"eval_spike_ep{ep}_step{max_spike_idx}_td{max_td_val:.2f}.mp4"
                out_path = os.path.join(out_dir, filename)
                
                with imageio.get_writer(
                    out_path, fps=5, codec="libx264", macro_block_size=None, ffmpeg_params=["-pix_fmt", "yuv420p"]
                ) as writer:
                    for fr in frames:
                        writer.append_data(fr)
                print(f"🎬 Video saved at: {out_path}\n")
        else:
            print(f"[Ep {ep+1}/{num_episodes}] No significant TD spikes. (Max TD: {np.max(td_array):.2f})\n")

    env.close()
    ray.shutdown()

if __name__ == "__main__":
    CHECKPOINT_DIR = "/home/jsr/project/Overcooked-ai/ppo_agent/results/PPO_Overcooked_Shared_asymmetric_advantages_reward_shaping_5e-6/PPO_overcooked_ppo_shared_reward_shaping_5e-6_fc64_93743_00000_0_2026-03-20_23-32-08/checkpoint_000029" 
    
    evaluate_and_record_td_spikes(
        checkpoint_path=CHECKPOINT_DIR,
        layout_name="asymmetric_advantages",
        num_episodes=10,        
        n_steps=15,             # 에러 발생 기준 앞뒤 15스텝 캡처
        z_threshold=3.0,        # 평균 대비 3표준편차 이상 튈 때 감지
        min_td_threshold=1.0    
    )
