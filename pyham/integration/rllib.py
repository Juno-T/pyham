try:
  import ray
  import ray.rllib
except:
  raise("Installation needed for rllib integration.")

import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks

class UploadCheckpointCallbacks(ray.tune.Callback):
  def on_checkpoint(self, iteration, trials, trial, checkpoint, **info):
    # try:
    print(trial.trial_id)
    # checkpoint.commit(os.path.join(wandb.run.dir, 'models/', 'checkpoint'))
    # except:
    #   print("No wandb run")
    #   print(checkpoint)

class CustomMetricAgentCallback(DefaultCallbacks):
  def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
    eval = worker.env_context.get("eval", False)
    print("\n\n\n")
    print(eval)
    print(episode.total_reward)
    print(episode.agent_rewards)
    print("\n\n\n")