from datetime import datetime
import gym
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.rllib.env.env_context import EnvContext
import argparse
from pyvirtualdisplay import Display

from pyham.ham.wrapper import create_concat_joint_state_wrapped_env

from machines import create_trivial_cartpole_ham, create_balance_recover_cartpole_ham

def make_wrapped_env(config: EnvContext):
  eval = config.get("eval", False) # bool # eval=True->will_render
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
                            will_render=False
                            # will_render=eval
                            )
  return wrapped_env

def main(config):
  wandb_init = {
      "project":"pyham-rllib-example",
      "tags": [config["machine_type"], config["env_name"], "rllib_cartpole", config["algo"]],
      "log_config": False,
      "api_key_file": "~/.wandb_api_key",
      "group": config["group"],

      # Callback args
      "save_checkpoints": config["save_checkpoints"],
  }

  train_config = {
    "seed": config["seed"],
    # ENV
    "env": config["env"],
    "horizon": config["horizon"],
    "env_config": config["env_config"],

    # AGENT
    "lr": config["learning_rate"],
    "gamma": config["discount"],
    "train_batch_size": config["train_batch_size"],
    
    # EXP
    "evaluation_interval": config["evaluation_interval"],
    "evaluation_duration": config["evaluation_duration"],
    "evaluation_config": {
      "env_config": {
        **config["env_config"],
        "eval": True,
      },
      "explore": False,
    },

    # SPEC
    "num_gpus": 0.2,
    "framework": "torch"
  }
  try:
    ray.init(address='auto')
  except:
    ray.init()
  tune.run(
    config["algo"],
    stop={"num_env_steps_trained": config["total_timesteps"]},
    config=train_config,
    num_samples=config["num_samples"],
    checkpoint_at_end=config["save_checkpoints"],
    # keep_checkpoints_num=1, 
    # checkpoint_score_attr="evaluation/episode_reward_mean",
    checkpoint_freq=config["evaluation_interval"],
    callbacks=[WandbLoggerCallback(**wandb_init)]
  )

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--machine',
                    default='balance-recover',
                    const='balance-recover',
                    nargs='?',
                    choices=['trivial', 'balance-recover'],
                    help="either trivial or balance-recover")
  parser.add_argument('--trial-number', type=int, default=-1,
                      help='trial number')
  parser.add_argument('--rep', type=int, default=0,
                      help='replication (ray\'s num_samples')
  parser.add_argument('--group', type=str, default=None,
                      help='wandb group')
  parser.add_argument('--save-checkpoints', type=str, default=False,
                      help='Save checkpoints')
  parser.add_argument('--hparam-search',
                    action='store_true',
                    default=False,
                    help="Run hyperparameter search")
  args = parser.parse_args()

  # default config
  config = {
    # Environment config
    "env_config":{
      "env_name": "CartPole-v1",
      "machine_type": str(args.machine), # "balance-recover"
      "machine_stack_cap": 1,
      "machine_stack_padding_value": 0,
      "internal_discount": 1,
    },
    "horizon": 500,

    # Agent config
    "algo": "PPO",
    "learning_rate": 1e-4,
    "discount": 0.99,
    "train_batch_size": 4000, # default

    # Experiment config
    "total_timesteps": 30000, # 4000 ts/iter
    "evaluation_interval_timesteps": 10000,
    "evaluation_duration": 5, # episodes
    "num_samples": args.rep if args.rep>0 else 1, # 1
    
    # wandb
    "save_checkpoints": args.save_checkpoints, # False # save ckp and upload to wandb
    "group": args.group # None
  }

  # process args
  config["env"] = f"{config['env_config']['env_name']}_{config['env_config']['machine_type']}"
  config["seed"] = args.trial_number if args.trial_number!=-1 else None
  if args.hparam_search:
    config["learning_rate"] = tune.loguniform(1e-5,1e-2)
    config["train_batch_size"] = tune.lograndint(128, 1024)
    config["num_samples"] = 10 if args.rep<1 else args.rep # number of trials
    config["save_checkpoints"] = False

  if config["num_samples"]>1:
    now = datetime.now()
    config["group"] = config["group"] or config["env"]+"_"+now.strftime("%b-%d-%Y,%H.%M.%S")

  # post-process config
  try:
    config["evaluation_interval"] = config["evaluation_interval_timesteps"]//config["train_batch_size"]
  except:
    config["evaluation_interval"] = tune.sample_from(
      lambda spec: config["evaluation_interval_timesteps"]//spec.config.train_batch_size
    )
  config = {**config, **config["env_config"]}

  # Run exp
  register_env(config["env"], make_wrapped_env)
  with Display(visible=False, size=(1400, 900)) as disp:
    main(config)
  

