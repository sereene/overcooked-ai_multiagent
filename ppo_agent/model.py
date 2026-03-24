import torch
import torch.nn as nn
import numpy as np
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork

import torch
import torch.nn as nn
import numpy as np
# 👇 RecurrentNetwork 대신 TorchModelV2를 임포트합니다.
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class CustomModel(TorchModelV2, nn.Module):
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
        
        # 3. 출력 헤드
        self.policy_head = nn.Linear(hidden_dim, num_outputs)
        
        # QMIX 에이전트 모델은 value_head가 필요 없으므로 삭제했습니다.
        self.hidden_dim = hidden_dim

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
        
        logits = self.policy_head(x) 
        
        return logits, state
    

