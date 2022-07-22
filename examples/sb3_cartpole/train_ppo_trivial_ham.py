import os
import gym
from stable_baselines3 import PPO
import wandb
from wandb.integration.sb3 import WandbCallback
  

from pyham.ham import create_concat_joint_state_wrapped_env
from pyham.examples.utils import EvalAndRenderCallback

from machines import create_trivial_cartpole_ham

def make_wrapped_env(config, eval=False):
  original_env = gym.make(config["env_name"], new_step_api=False)
  ham, choice_space, initial_machine, initial_args = create_trivial_cartpole_ham(config["internal_discount"])

  wrapped_env = create_concat_joint_state_wrapped_env(ham, 
                            original_env, 
                            choice_space, 
                            initial_machine=initial_machine,
                            initial_args=initial_args,
                            np_pad_config = {"constant_values": config["machine_stack_padding_value"]},
                            machine_stack_cap=config["machine_stack_cap"],
                            will_render=eval)
  return wrapped_env

def main():
  config = {
    "internal_discount": 0.95,
    "machine_stack_cap": 1,
    "machine_stack_padding_value": 0,
    "policy_type": "MlpPolicy",
    "total_timesteps": 25000,
    "env_name": "CartPole-v1",
    "n_eval_episodes": 5,
  }

  run = wandb.init(
      resume="allow",
      tags=["trivial machine","ppo", "cartpole"],
      project="sb3",
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
                  render_freq=2500
                ),
              ],
  )
  wrapped_env.close()
  wrapped_env_eval.close()
  run.finish()


if __name__=="__main__":
  main()