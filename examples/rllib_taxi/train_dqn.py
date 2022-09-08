from datetime import datetime
from math import ceil
import argparse
import yaml
import json
import gym
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.rllib.env.env_context import EnvContext
from ray.rllib.policy.policy import PolicySpec
from pyvirtualdisplay import Display

from pyham.wrappers.helpers import create_concat_joint_state_env
from pyham.integration.gym_wrappers import DecodedMultiDiscreteWrapper
from pyham.utils import deep_dict_update

from machines import create_taxi_ham, create_trivial_taxi_ham


def make_taxi_env():
  env = gym.make("Taxi-v3") # Already have time limit = 200
  env = DecodedMultiDiscreteWrapper(env, [5,5,5,4])
  return env

def make_wrapped_env(config: EnvContext):
  eval = config.get("eval", False)
  original_env = make_taxi_env()

  create_machine = None
  if config["machine_type"]=="trivial":
    create_machine = create_trivial_taxi_ham
  elif config["machine_type"]=="get-put":
    create_machine = create_taxi_ham
  ham, initial_machine, initial_args = create_machine()

  kwargs = {
    "initial_machine": initial_machine,
    "initial_args": initial_args,
    "np_pad_config": {"constant_values": config["machine_stack_padding_value"]},
    "machine_stack_cap": config["machine_stack_cap"],
    "eval": eval,
    "will_render": False,
  }

  wrapped_env = create_concat_joint_state_env(ham, 
                                              original_env,
                                              **kwargs)
  return wrapped_env

def extra_choicepoint_config(machine_type):
  if machine_type=="trivial":
    return {
      "original_choice": {},
    }
  elif machine_type=="get-put":
    return {
      "get_put": {},
      "nav": {},
    }

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
    # "model": config["model"],
    "exploration_config": config["exploration_config"],
    "rollout_fragment_length": config["rollout_fragment_length"],
    "dueling": config["dueling"], # DQN
    "replay_buffer_config": config["replay_buffer_config"], # DQN
    "training_intensity": config["training_intensity"], # DQN

    # SPEC
    "num_workers": 3,
    "num_gpus": 0.25,
    "framework": "torch"
  }
  if "multiagent" in config:
    train_config["multiagent"] = config["multiagent"]
  try:
    ray.init(address='auto')
  except:
    print("Start new ray instance\n")
    ray.init(num_cpus=16)
  tune.run(
    config["algo"],
    stop={"timesteps_total": config["total_timesteps"]},
    config=train_config,
    num_samples=config["num_samples"],
    checkpoint_at_end=config["save_checkpoints"],
    callbacks=[WandbLoggerCallback(**wandb_init)]
  )

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
      "env_name": "Taxi-v3",
      "machine_type": str(args.machine), # "get-put"
      "machine_stack_cap": 3 if str(args.machine)=="get-put" else 1,
      "machine_stack_padding_value": 0,
    },
    "horizon": 200,

    # Default Agent config
    "algo": "DQN",
    "learning_rate": 3e-3,
    "discount": 0.99,
    "rollout_fragment_length": 200,
    "train_batch_size": 200, # default
    "exploration_config": {
      "type": "EpsilonGreedy",
      "warmup_timesteps": 40000,
      "epsilon_timesteps": int(400000*0.5)-40000, # 50% explore fraction
    },
    "dueling": True, # DQN
    "replay_buffer_config": { # DQN
      "capacity": 200000,
    },
    "training_intensity": 8, # DQN (=Sample efficiency)

    # Experiment config
    "total_timesteps": 400000, 
    # "evaluation_interval_timesteps": 5000,
    # "evaluation_duration": 5, # episodes
    "num_samples": args.rep if args.rep>0 else 1, # 1
    
    # wandb
    "save_checkpoints": args.save_checkpoints, # False # save ckp and upload to wandb
    "group": args.group, # None

    # extra
    "exploration_fraction": 0.5
  }

  # process args
  config["env"] = f"{config['env_config']['env_name']}_{config['env_config']['machine_type']}"
  config["seed"] = args.trial_number if args.trial_number!=-1 else None
  if args.hparam_search:
    config["learning_rate"] = tune.loguniform(1e-5,3e-2)
    config["train_batch_size"] = tune.lograndint(200, 1000)
    config["training_intensity"] = tune.lograndint(1,25)
    config["exploration_config"]["epsilon_timesteps"] = tune.randint(40000,320000) # 20%-90%
    config["num_samples"] = 10 if args.rep<1 else args.rep # number of trials
    config["save_checkpoints"] = False

  if config["num_samples"]>1:
    now = datetime.now()
    config["group"] = config["group"] or config["env"]+"_"+now.strftime("%b-%d-%Y,%H.%M.%S")
  else:
    config["group"] = config["group"] or "Taxi-v3"

  # Multi-agent config
  if config["env_config"]["machine_type"]!="trivial":
    tmp_env = make_wrapped_env(config["env_config"])
    policies, policy_mapping_fn, policies_to_train = tmp_env.rllib_policies_config()
    extra_config = extra_choicepoint_config(config["env_config"]["machine_type"])
    config["multiagent"]= {
      "count_steps_by": "agent_steps",
      "policies": {
          cp_name: PolicySpec(**dict(cp_config, 
            config=dict(cp_config.get("config", {}), **extra_config.get(cp_name, {}))
          ))
        for cp_name, cp_config in policies.items()  
      },
      "policy_mapping_fn": policy_mapping_fn,
      "policies_to_train": policies_to_train
    }
  
  # post-process config
  if not args.hparam_search:
    if config["exploration_fraction"] is not None:
      config["exploration_config"]["epsilon_timesteps"] = int(config["total_timesteps"]*config["exploration_fraction"] - config["exploration_config"]["warmup_timesteps"])

    # Tuned config
    with open(f'examples/rllib_taxi/config_{args.machine}.yaml') as f:
      tuned_config = yaml.safe_load(f)["config"]
    config = deep_dict_update(config, tuned_config)

  config = {**config, **config["env_config"]}
  print("Config:\n",json.dumps(config, sort_keys=False, indent=2, default=str))

  # Run exp
  register_env(config["env"], make_wrapped_env)
  with Display(visible=False, size=(1400, 900)) as disp:
    main(config)
