import gymnasium as gym
import numpy as np
import pygame
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from ray.rllib.env.multi_agent_env import MultiAgentEnv

ACTION_MAP = {
    0: (0, -1),   # NORTH
    1: (0, 1),    # SOUTH
    2: (1, 0),    # EAST
    3: (-1, 0),   # WEST
    4: (0, 0),    # STAY
    5: "interact" # INTERACT
}

class Rllib_multi_agent(MultiAgentEnv):
    def __init__(self, config = None):
        #이후 config에 layout name, horizon 등등을 넣어야함.
        super().__init__()
        config = config or {}
        layout_name = config.get("layout_name", "cramped_room")
        horizon = config.get("horizon", 400)

        mdp = OvercookedGridworld.from_layout_name(layout_name) 
        self.overcooked_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
        
        self.agents = ["agent_0", "agent_1"]
        self._agent_ids = {"agent_0", "agent_1"}
        #self._agent_ids = set(self.agents)
        sample_obs_dict = self._get_obs(0)
        flattened_shape = sample_obs_dict['agent_0'].flatten().shape
        
        self.observation_space = gym.spaces.Dict(
            {
                "agent_0": gym.spaces.Box(low=-np.inf, high=np.inf, shape=flattened_shape, dtype=np.float32),
                "agent_1": gym.spaces.Box(low=-np.inf, high=np.inf, shape=flattened_shape, dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Dict(
            {
                "agent_0": gym.spaces.Discrete(len(ACTION_MAP)),
                "agent_1": gym.spaces.Discrete(len(ACTION_MAP)),
            }
        )
        
        self.visualizer = StateVisualizer()

    # def __init__(self, config=None):
    #     super().__init__()
    #     config = config or {}
    #     self.layout_name = config.get("layout_name", "asymmetric_advantages")
    #     horizon = config.get("horizon", 400)
    #     self.use_reward_shaping = config.get("reward_shaping", False)

    #     mdp = OvercookedGridworld.from_layout_name(self.layout_name) 
    #     self.overcooked_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
        
    #     self.agents = ["agent_0", "agent_1"]
    #     self._agent_ids = set(self.agents)

    #     dummy_state = self.overcooked_env.mdp.get_standard_start_state()
        
    #     # 🔥 [핵심 수정] 끝에 있던 [0]을 제거했습니다! 
    #     # 이제 두 에이전트의 데이터가 정상적으로 모두 들어옵니다.
    #     sample_obs_list = self.overcooked_env.featurize_state_mdp(dummy_state)
        
    #     # 96차원 배열을 정상적으로 가져옵니다.
    #     sample_obs = np.array(sample_obs_list[0], dtype=np.float32).flatten()
    #     feature_length = sample_obs.shape[0]  
        
    #     # (1, 96) 형태로 설정하여 QMIX의 AxisError 버그까지 완벽 차단!
    #     self.observation_space = gym.spaces.Box(
    #         low=-np.inf, high=np.inf, shape=(1, feature_length), dtype=np.float32
    #     )
    #     self.action_space = gym.spaces.Discrete(len(ACTION_MAP))
    #     self.visualizer = StateVisualizer()


    def _get_obs(self, idx = 0):
        state = self.overcooked_env.state

        #print(state.shape)
        obs_tuple = self.overcooked_env.lossless_state_encoding_mdp(state)
        observations = {
            self.agents[0]: obs_tuple[0].flatten().astype(np.float32),
            self.agents[1]: obs_tuple[1].flatten().astype(np.float32),
        }
        #obs1 = np.array(obs).flatten()
        return observations
    
    # def _get_obs(self):
    #     state = self.overcooked_env.state
        
    #     obs_list = self.overcooked_env.featurize_state_mdp(state)
        
    #     observations = {
    #         self.agents[0]: np.expand_dims(np.array(obs_list[0], dtype=np.float32), axis=0),
    #         self.agents[1]: np.expand_dims(np.array(obs_list[1], dtype=np.float32), axis=0),
    #     }
    #     return observations

    #obs, reward 공유됨.
    def reset(self, seed=None, options=None):
        """환경을 리셋하고 각 에이전트의 초기 관측값을 반환합니다."""
        self.overcooked_env.reset()
        #self.agents = ["agent_1", "agent_2"]
        # 각 에이전트 ID에 대한 관측값을 담은 딕셔너리를 반환합니다.
        obs = self._get_obs()
        return obs, {}

    def step(self, action_dict):

        #print("Received action_dict:", action_dict)
        # if action_dict == {}:
        #     action_dict['agent_1'] = 4
        #     action_dict['agent_2'] = 4
        #     print(1)
            
        actions = [ACTION_MAP[action_dict[agent_id]] for agent_id in self.agents]

        next_state, rewards, done, info = self.overcooked_env.step(actions)
        obs = self._get_obs()
        shaped_rewards_list = info["shaped_r_by_agent"]

        info["sparse_reward"] = float(rewards)

        #if shaped_rewards_list[0] !=0 or shaped_rewards_list[1] != 0:
        #    print(shaped_rewards_list[0], shaped_rewards_list[1])
        reward = {
            self.agents[0]: rewards + shaped_rewards_list[0],
            self.agents[1]: rewards + shaped_rewards_list[1],         
        }

        done_dict = {
            self.agents[0]: done,
            self.agents[1]: done,  
        }
        truncated_dict = {
            self.agents[0]: False,
            self.agents[1]: False,  
        }
        done_dict["__all__"] = done
        truncated_dict["__all__"] = False  

        info_dict = {
            self.agents[0]: info,
            self.agents[1]: info, 
        }

        return obs, reward, done_dict, truncated_dict, info_dict

        info_dict = {
            self.agents[0]: info,
            self.agents[1]: info, 
        }

        return obs, reward, done_dict, truncated_dict, info_dict


    def render(self, mode="rgb_array"):
        # callbacks.py에서 MP4 영상을 추출할 수 있도록 RGB 형태의 프레임을 반환합니다.
        surface = self.visualizer.render_state(
            state=self.overcooked_env.state,
            grid=self.overcooked_env.mdp.terrain_mtx
        )
        # Pygame surface를 Numpy 배열(RGB)로 변환
        frame = pygame.surfarray.pixels3d(surface)
        frame = np.transpose(frame, (1, 0, 2))  # (W, H, C) -> (H, W, C)
        
        # 👇 추가된 부분: FFMPEG 인코딩 에러(Broken Pipe) 방지를 위한 짝수 해상도 보정
        h, w, c = frame.shape
        if h % 2 != 0:
            frame = frame[:-1, :, :]  # 세로가 홀수면 마지막 1픽셀 제거
        if w % 2 != 0:
            frame = frame[:, :-1, :]  # 가로가 홀수면 마지막 1픽셀 제거
            
        return frame.astype(np.uint8) # 안전하게 uint8 타입으로 변환하여 반환