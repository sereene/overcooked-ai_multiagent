import torch
import torch.nn as nn
import numpy as np
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork

import torch
import torch.nn as nn
import numpy as np
# 👇 RecurrentNetwork 대신 TorchModelV2를 임포트합니다.
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class CustomQMIXModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # 👇 초기화도 TorchModelV2에 맞게 변경
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # train.py의 "custom_model_config"에서 전달받은 파라미터 가져오기
        custom_config = model_config.get("custom_model_config", {})
        hidden_dim = custom_config.get("fc_size", 64)

        # 관측 공간(obs_space)의 1D 크기 자동 계산
        input_dim = int(np.prod(obs_space.shape))

        # 1. Flatten MLP (특징 추출기)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 2. GRU
        self.gru = nn.GRU(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        
        # 3. 출력 헤드
        self.policy_head = nn.Linear(hidden_dim, num_outputs)
        
        # QMIX 에이전트 모델은 value_head가 필요 없으므로 삭제했습니다.
        self.hidden_dim = hidden_dim

    # --- QMIX 필수 구현 메서드 1: 초기 상태 ---
    def get_initial_state(self):
        return [torch.zeros(self.hidden_dim, dtype=torch.float32)]

    # --- QMIX 필수 구현 메서드 2: 포워드 ---
    def forward(self, input_dict, state, seq_lens):
        # QMIX는 딕셔너리가 아닌 텐서 자체를 넘기기도 하므로 방어적으로 처리합니다.
        if isinstance(input_dict, dict):
            obs = input_dict.get("obs_flat", input_dict.get("obs"))
        else:
            obs = input_dict
            
        # 강제로 2D (Batch, Features) 형태로 평탄화
        if len(obs.shape) > 2:
            obs = obs.reshape(obs.shape[0], -1)

        # MLP 통과
        x = self.mlp(obs.float()) 
        
        # GRU 입력을 위한 차원 추가: (Batch, Seq_Len=1, Features)
        x = x.unsqueeze(1)
        
        # GRU Hidden State 형태 맞추기: (num_layers=1, Batch, Features)
        h_in = state[0].unsqueeze(0)
        
        # GRU 통과
        gru_out, h_out = self.gru(x, h_in)
        
        # Q-values (Logits) 산출을 위한 차원 축소: (Batch, Features)
        gru_out = gru_out.squeeze(1)
        logits = self.policy_head(gru_out) 
        
        # 반환값: (로짓, [업데이트된 상태 리스트])
        return logits, [h_out.squeeze(0)]
    


# import torch
# import torch.nn as nn
# import numpy as np

# class CustomQMixer(nn.Module):
#     def __init__(self, n_agents, state_shape, mixing_embed_dim, fc_size=64):
#         super(CustomQMixer, self).__init__()

#         self.n_agents = n_agents
#         self.embed_dim = mixing_embed_dim
        
#         # RLlib이 전달하는 Flatten된 원본 State의 길이
#         self.state_dim = int(np.prod(state_shape))

#         # ==========================================
#         # [수정된 부분] 1. CNN 대신 MLP + GRU 구조 적용
#         # ==========================================
#         self.hidden_dim = fc_size
        
#         # 특징 추출기 (MLP)
#         self.state_mlp = nn.Sequential(
#             nn.Linear(self.state_dim, self.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.hidden_dim, self.hidden_dim),
#             nn.ReLU()
#         )
        
#         # ==========================================
#         # 2. 글로벌네트워크 (원본 그대로 유지)
#         # ==========================================
#         self.hyper_w_1 = nn.Linear(self.hidden_dim, self.embed_dim * self.n_agents)
#         self.hyper_w_final = nn.Linear(self.hidden_dim, self.embed_dim)
#         self.hyper_b_1 = nn.Linear(self.hidden_dim, self.embed_dim)
        
#         self.V = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.embed_dim),
#             nn.ReLU(),
#             nn.Linear(self.embed_dim, 1),
#         )

#     def forward(self, agent_qs, states):
#         """Forward pass for the mixer."""
#         bs = agent_qs.size(0)
        
#         # 1. RLlib이 주는 1차원 배열을 가져옴
#         states = states.reshape(-1, self.state_dim).float()

#         # ==========================================
#         # 2. MLP를 거쳐 state_emb 추출 (GRU 제거)
#         # ==========================================
#         state_emb = self.state_mlp(states)

#         # ==========================================
#         # 3. 원본 믹싱 연산 (추출된 state_emb 사용)
#         # ==========================================
#         agent_qs = agent_qs.view(-1, 1, self.n_agents)
        
#         # First layer
#         w1 = torch.abs(self.hyper_w_1(state_emb))
#         b1 = self.hyper_b_1(state_emb)
#         w1 = w1.view(-1, self.n_agents, self.embed_dim)
#         b1 = b1.view(-1, 1, self.embed_dim)
#         hidden = nn.functional.elu(torch.bmm(agent_qs, w1) + b1)
        
#         # Second layer
#         w_final = torch.abs(self.hyper_w_final(state_emb))
#         w_final = w_final.view(-1, self.embed_dim, 1)
        
#         # State-dependent bias
#         v = self.V(state_emb).view(-1, 1, 1)
        
#         # Compute final output
#         y = torch.bmm(hidden, w_final) + v
        
#         # Reshape and return
#         q_tot = y.view(bs, -1, 1)
#         return q_tot