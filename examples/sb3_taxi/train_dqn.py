import os
import gym
import argparse
from stable_baselines3 import DQN
import wandb
from wandb.integration.sb3 import WandbCallback
  
from pyham.ham import create_concat_joint_state_wrapped_env
from pyham.examples.utils import EvalAndRenderCallback
from pyham.examples.wrappers import DecodedMultiDiscreteWrapper, MultiDiscrete2NormBoxWrapper

from machines import create_taxi_ham, create_trivial_taxi_ham

def make_env(config):
  assert(config["env_name"] == "Taxi-v3")
  env = gym.make(config["env_name"], new_step_api=False) # Already have time limit = 200
  env = DecodedMultiDiscreteWrapper(env, [5,5,5,4])
  # env = MultiDiscrete2NormBoxWrapper(env)
  return env

def make_wrapped_env(config, eval=False):
  original_env = make_env(config)
  create_machine = None
  if config["machine_type"]=="trivial":
    create_machine = create_trivial_taxi_ham
  elif config["machine_type"]=="get-put":
    create_machine = create_taxi_ham
  ham, choice_space, initial_machine, initial_args = create_machine(config["internal_discount"])

  wrapped_env = create_concat_joint_state_wrapped_env(ham, 
                            original_env, 
                            choice_space, 
                            initial_machine=initial_machine,
                            initial_args=initial_args,
                            np_pad_config = {"constant_values": config["machine_stack_padding_value"]},
                            machine_stack_cap=config["machine_stack_cap"],
                            will_render=eval)
  return wrapped_env

def main(config):

  run = wandb.init(
      tags=[config["machine_type"], "sb3_taxi", "dqn", "taxi"],
      project="pyham-example",
      config=config,
      sync_tensorboard=True,
  )
  run.config.setdefaults(config)
  config=wandb.config

  
  wrapped_env = make_wrapped_env(config)
  wrapped_env_eval = make_wrapped_env(config)

  model = DQN(config['policy_type'], 
              wrapped_env, 
              learning_rate=config["learning_rate"], 
              learning_starts=config["learning_starts"],
              exploration_fraction=config["exploration_fraction"],
              buffer_size=config["buffer_size"],
              verbose=1, 
              tensorboard_log=f"runs/{run.id}",
              seed=config["seed"]
            )
  model.learn(total_timesteps=config['total_timesteps'],
              callback=[
                WandbCallback(
                  gradient_save_freq=config["gradient_save_freq"],
                  model_save_path=f"models/{run.id}",
                  verbose=2,
                ),
                EvalAndRenderCallback(
                  wrapped_env_eval, 
                  n_eval_episodes=config["n_eval_episodes"], 
                  eval_freq=config["eval_freq"], 
                  render_freq=config["render_freq"],
                  fps=5,
                ),
              ],
            )
  wrapped_env.close()
  wrapped_env_eval.close()
  run.finish()


if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--machine',
                    default='get-put',
                    const='get-put',
                    nargs='?',
                    choices=['trivial', 'get-put'],
                    help="either trivial or get-put")
  parser.add_argument('--trial-number', type=int, default=-1,
                      help='trial number')
  args = parser.parse_args()

  config = {
    "learning_rate": 3e-3,
    "learning_starts": 200000,
    "exploration_fraction": 0.43,
    "buffer_size": 200000,
    "eval_freq": 100000,
    "render_freq": 500000,
    "gradient_save_freq": 500000,
    "internal_discount": 1,
    "machine_type": None, # "trivial" or "get-put"
    "machine_stack_cap": 1,
    "machine_stack_padding_value": 0,
    "policy_type": "MlpPolicy",
    "total_timesteps": 1500000,
    "env_name": "Taxi-v3",
    "n_eval_episodes": 5
  }
  config["machine_type"] = str(args.machine)
  if config["machine_type"] == "get-put":
    config["machine_stack_cap"] = 2
    config["learning_rate"] = 0.002306
  config["trial_number"] = int(args.trial_number)
  config["seed"] = config["trial_number"] if config["trial_number"]!=-1 else None
  main(config)