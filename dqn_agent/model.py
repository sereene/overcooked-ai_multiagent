import torch
import torch.nn as nn
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class CustomModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom_config = model_config.get("custom_model_config", {})
        hidden_dim = custom_config.get("fc_size", 64)

        input_dim = int(np.prod(obs_space.shape))

        # 1. Flatten MLP (특징 추출기)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 2. 출력 헤드
        self.policy_head = nn.Linear(hidden_dim, num_outputs)

    # 🔥 get_initial_state() 삭제됨

    def forward(self, input_dict, state, seq_lens):
        if isinstance(input_dict, dict):
            obs = input_dict.get("obs_flat", input_dict.get("obs"))
        else:
            obs = input_dict
            
        if len(obs.shape) > 2:
            obs = obs.reshape(obs.shape[0], -1)

        x = self.mlp(obs.float()) 
        logits = self.policy_head(x) 
        
        # RNN을 사용하지 않으므로 들어온 빈 state 리스트를 그대로 반환
        return logits, state