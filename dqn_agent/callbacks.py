import os
import gc
import numpy as np
import imageio.v2 as imageio
from datetime import datetime
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import wandb

RUN_TS_ENV = "VIDEO_RUN_TS"

def _get_run_timestamp() -> str:
    ts = os.environ.get(RUN_TS_ENV)
    if not ts:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.environ[RUN_TS_ENV] = ts
    return ts

class OvercookedCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        episode.user_data["sparse_reward"] = 0.0

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        info = episode.last_info_for("agent_0")
        if info and "sparse_reward" in info:
            episode.user_data["sparse_reward"] += info["sparse_reward"]

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        total_score = episode.total_reward
        episode.custom_metrics["score"] = total_score
        
        sparse_score = episode.user_data.get("sparse_reward", 0.0)
        episode.custom_metrics["sparse_score"] = sparse_score
        
def rollout_and_save_video(
    *,
    algorithm,
    out_path: str,
    env_creator_fn,
    max_cycles: int = 500,
    every_n_steps: int = 1, 
    max_frames: int = 500,
    fps: int = 10,  
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    env = env_creator_fn({"layout_name": "asymmetric_advantages"})
    frames = []

    try:
        obs, infos = env.reset()
        step_i = 0

        def get_frame():
            target_env = env.env if hasattr(env, "env") else env
            if hasattr(target_env, "render"):
                return target_env.render()
            return None

        fr0 = get_frame()
        if fr0 is not None:
            frames.append(fr0)

        while True:
            if step_i >= max_cycles:
                break

            actions = {}
            
            # 🔥 DQN은 RNN 상태(state)가 필요 없으므로 단순화됨
            for agent_id, agent_obs in obs.items():
                action = algorithm.compute_single_action(
                    agent_obs,
                    policy_id="shared_policy", 
                    explore=False,
                )
                actions[agent_id] = action

            obs, rewards, terminations, truncations, infos = env.step(actions)

            if (step_i % every_n_steps) == 0:
                if len(frames) >= max_frames:
                    break
                fr = get_frame()
                if fr is not None:
                    frames.append(fr)

            step_i += 1

            if terminations.get("__all__", False) or truncations.get("__all__", False) or len(obs) == 0:
                break

        if not frames:
            print("[VIDEO] No frames captured (env.render() returned None).")
            return

        safe_frames = []
        for fr in frames:
            if fr is None:
                continue
            if isinstance(fr, np.ndarray):
                if fr.dtype != np.uint8:
                    fr = np.clip(fr, 0, 255).astype(np.uint8)
                safe_frames.append(fr)

        if not safe_frames:
            print("[VIDEO] Frames existed but none were valid numpy arrays.")
            return

        with imageio.get_writer(
            out_path,
            fps=fps,
            codec="libx264",
            macro_block_size=None,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        ) as writer:
            for fr in safe_frames:
                writer.append_data(fr)

        print(f"[VIDEO] saved locally: {out_path} ({len(safe_frames)} frames)")

        if wandb.run is not None:
            try:
                wandb.log(
                    {
                        "evaluation/gameplay_video": wandb.Video(
                            out_path,
                            fps=fps,
                            format="mp4",
                            caption=f"Eval iter={algorithm.training_iteration}",
                        )
                    },
                    step=int(algorithm.training_iteration),
                    commit=True,
                )
            except Exception as e:
                print(f"[WandB] Video upload failed: {e}")

    finally:
        try:
            env.close()
        except Exception:
            pass
        gc.collect()

class VideoCallbacks(OvercookedCallbacks):
    def __init__(self, out_dir: str, env_creator_fn, every_n_evals: int = 5, max_cycles: int = 400):
        super().__init__()
        self.base_out_dir = out_dir
        self.run_ts = _get_run_timestamp()
        
        self.run_dir = os.path.join(self.base_out_dir, self.run_ts)
        os.makedirs(self.run_dir, exist_ok=True)

        self.env_creator_fn = env_creator_fn
        self.every_n_evals = every_n_evals
        self.max_cycles = max_cycles
        self.eval_count = 0
        self._last_saved_iter = -1

    def on_train_result(self, *, algorithm, result, **kwargs):
        if "evaluation" not in result:
            return

        self.eval_count += 1
        if (self.eval_count % self.every_n_evals) != 0:
            return

        training_iter = int(result.get("training_iteration", 0))

        if training_iter == self._last_saved_iter:
            return
        self._last_saved_iter = training_iter

        video_filename = f"eval_{self.eval_count:04d}_iter{training_iter:06d}.mp4"
        out_path = os.path.join(self.run_dir, video_filename)

        rollout_and_save_video(
            algorithm=algorithm,
            out_path=out_path,
            env_creator_fn=self.env_creator_fn,
            max_cycles=self.max_cycles,
            fps=10, 
        )