import os
from pathlib import Path
import numpy as np
import cv2
from stable_baselines3.common.callbacks import BaseCallback

try:
    import wandb
except ImportError:
    wandb = None

class EvalAndRenderCallback(BaseCallback):
  """
    SB3 Callback for evaluating and rendering an agent.
    wandb is required to see the logs and rendering.
  """
  def __init__(self, eval_env, n_eval_episodes=5, eval_freq=20, render_freq=2500):
    """
      Parameters:
        eval_env: (gym.Env) The environment used for initialization
        n_eval_episodes: (int) The number of episodes to test the agent
        eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
        render_freq: (int) Render an episode every render_freq call of the callback.
    """
    super(EvalAndRenderCallback, self).__init__()
    self.eval_env = eval_env
    self.n_eval_episodes = n_eval_episodes
    self.eval_freq = eval_freq
    self.render_freq = render_freq
    self.wandb = wandb

    self.best_mean_reward = -np.inf
  
  def _on_step(self):
    wandb_log_dict = {"trained_step": self.n_calls}
    if self.n_calls % self.eval_freq == 0:
      avg_reward = self.play(episode = self.n_eval_episodes)
      new_best = False
      if avg_reward > self.best_mean_reward:
        new_best = True
        self.best_mean_reward=avg_reward
      
      if self.wandb:
        wandb_log_dict["eval/ep_rew_mean"] = avg_reward
        if new_best:
          self.model.save(os.path.join(self.wandb.run.dir, 'models/', 'best.zip'))

      print("Best mean reward: {:.2f}".format(self.best_mean_reward))
      
    if self.n_calls % self.render_freq == 0 and self.wandb:
      frames = self.render_play()
      wandb_log_dict["eval/render"] = wandb.Video(process_frames(frames), fps=15, format="mp4")

    if self.wandb:
      self.wandb.log(wandb_log_dict)
    

    return True

  def play(self, episode=1):
    # TODO log ep length
    self.eval_env.set_render_mode(False)
    rewards = []
    for i in range(episode):
      obs = self.eval_env.reset()
      done=False
      cumulative_reward = 0.
      while not done:
        action, _states = self.model.predict(obs)
        obs, reward, done, info = self.eval_env.step(action)
        cumulative_reward+=reward
      rewards.append(cumulative_reward)
    avg_reward = np.mean(rewards)
    return avg_reward

  def render_play(self):
    self.eval_env.set_render_mode(True)
    obs = self.eval_env.reset()
    frames = self.eval_env.render()
    done=False
    while not done:
      action, _states = self.model.predict(obs)
      obs, reward, done, info = self.eval_env.step(action)
      frames+=self.eval_env.render()
    self.eval_env.set_render_mode(False)
    return frames

def render_frames(frames, outpath, fps=30, quality = "low"):
  assert outpath[-4:]==".mp4", "Not supported path"
  outpath=Path(outpath)
  os.makedirs(outpath.parent, exist_ok=True)
  size = frames[0].shape
  frameskip=1 # all frame
  if quality=="low":
    size = [s//2 for s in size]
    fps = fps//2
    frameskip=2 # every other frame
  out = cv2.VideoWriter(str(outpath), cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]))
  print(f"Rendering video with {len(frames)} frames of shape {frames[0].shape}")
  for frame in frames[::frameskip]:
    frame = np.asarray(frame)
    if frame.shape!=size:
      frame = cv2.resize(frame, (size[1], size[0]))
    # print(frame.mean(), frame.shape)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out.write(frame)
  out.release()
  print(f"Rendered video at {str(outpath)}")


def process_frames(frames, quality = "low"):
  """
    Process frames to wandb applicable format
  """
  size = frames[0].shape
  frameskip=1 # all frame
  if quality=="low":
    size = [size[0]//2, size[1]//2, size[2]]
    frameskip=2 # every other frame
  new_frames = [cv2.resize(frame, (size[1], size[0])) for frame in frames[::frameskip]]
  new_frames = np.array(new_frames)
  new_frames = np.transpose(new_frames, (0,3,1,2))
  return new_frames