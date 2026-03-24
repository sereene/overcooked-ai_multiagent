import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

# 1. 맵 레이아웃 지정 (원하는 맵 이름으로 변경)
layout_name = "cramped_room"
mdp = OvercookedGridworld.from_layout_name(layout_name)
env = OvercookedEnv.from_mdp(mdp, horizon=400)

# 2. 게임의 초기 상태(State) 가져오기
state = env.mdp.get_standard_start_state()

# 3. 두 가지 방식의 관측값 추출
# [방식 A] Lossless State Encoding (맵 전체의 정보를 손실 없이 3D 그리드로 표현)
lossless_obs = env.lossless_state_encoding_mdp(state)

# [방식 B] Featurized State (거리, 방향 등 사람이 직접 설계한 특성들을 1D 벡터로 표현)
featurize_obs = env.featurize_state_mdp(state)

print(f"선택된 맵: {layout_name}\n")

print("=== 1. Lossless State Encoding (CNN 모델에 적합) ===")
lossless_np = np.array(lossless_obs)
print(f"-> 전체 배열 차원: {lossless_np.shape}  (에이전트수, X축, Y축, 채널수)")
print(f"-> 에이전트 1명의 원본 형태: {lossless_np[0].shape}")
print(f"-> 에이전트 1명을 Flatten(1차원) 했을 때 벡터 크기: {lossless_np[0].flatten().shape[0]}\n")

print("=== 2. Featurized State Encoding (MLP 모델에 적합) ===")
# featurize_state_mdp는 (관측값, info) 형태의 튜플을 반환하므로 [0]을 선택
featurize_np = np.array(featurize_obs[0]) 
print(f"-> 전체 배열 차원: {featurize_np.shape}  (에이전트수, 피처크기)")
print(f"-> 에이전트 1명의 1차원 벡터 크기: {featurize_np[0].shape[0]}")