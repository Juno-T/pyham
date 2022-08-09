import os
import gym
import argparse
from stable_baselines3 import PPO
import wandb
from wandb.integration.sb3 import WandbCallback
from pyvirtualdisplay import Display
  
from pyham.ham import create_concat_joint_state_wrapped_env
from pyham.examples.utils import EvalAndRenderCallback

from machines import create_trivial_cartpole_ham, create_balance_recover_cartpole_ham

def make_wrapped_env(config, eval=False):
  original_env = gym.make(config["env_name"])

  create_machine = None
  if config["machine_type"]=="trivial":
    create_machine = create_trivial_cartpole_ham
  elif config["machine_type"]=="balance-recover":
    create_machine = create_balance_recover_cartpole_ham
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
      tags=[config["machine_type"],"sb3_cartpole","ppo", "cartpole"],
      project="pyham-sb3-example",
      config=config,
      sync_tensorboard=True,
  )

  
  wrapped_env = make_wrapped_env(config)
  wrapped_env_eval = make_wrapped_env(config)
                        
  model = PPO(config['policy_type'], wrapped_env, verbose=1, tensorboard_log=f"runs/{run.id}")
  model.learn(total_timesteps=config['total_timesteps'],
              callback=[
                WandbCallback(
                  gradient_save_freq=1000,
                  model_save_path=f"models/{run.id}",
                  verbose=2,
                ),
                EvalAndRenderCallback(
                  wrapped_env_eval, 
                  n_eval_episodes=config["n_eval_episodes"], 
                  eval_freq=1000, 
                  render_freq=10000
                ),
              ],
  )
  wrapped_env.close()
  wrapped_env_eval.close()
  run.finish()


if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--machine',
                    default='balance-recover',
                    const='balance-recover',
                    nargs='?',
                    choices=['trivial', 'balance-recover'],
                    help="either trivial or balance-recover")
  args = parser.parse_args()

  config = {
    "internal_discount": 1,
    "machine_type": None, # "trivial" or "balance-recover"
    "machine_stack_cap": 1,
    "machine_stack_padding_value": 0,
    "policy_type": "MlpPolicy",
    "total_timesteps": 30000,
    "env_name": "CartPole-v1",
    "n_eval_episodes": 5,
  }
  config["machine_type"] = str(args.machine)
  with Display(visible=False, size=(1400, 900)) as disp:
    main(config)