# pyHAM: Python implementation of Hierarchies of Abstract Machine

:warning: This is not an official implementation.

**pyHAM** is a python implementation of Hierarchies of Abstract Machine, HAM, first introduced by [Parr and Russell (1997)](https://proceedings.neurips.cc/paper/1997/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf). The goal of this implementation is to make it easy for both researchers and practitioners to start developing reinforcement learning algorithm using HAM. Due to its simplicity for starting and integrating with other RL library, I hope that HAM and Hierarchical Reinforcement Learning (HRL) would receive more attention from the community.

[**:books: See wiki for more detailed documentation.**](https://github.com/Juno-T/pyham/wiki)


## installation
``` Shell
git clone https://github.com/Juno-T/pyham.git
export PYTHONPATH=$PWD:$PYTHONPATH
cd pyham
pip install -r requirements.txt             # requirements for core ham
pip install -r examples/requirements.txt    # requirements for RL library integration & running examples
```

## Running examples
``` Shell
wandb login
python3 examples/sb3_cartpole/train_ppo.py --machine trivial      # can replace trivial with balance-recover
python3 examples/sb3_taxi/train_dqn.py --machine trivial          # can replace trivial with get-put
```

## Quickstart
In this example, a HAM consisted of one machine and one choicepoint to control CartPole environment is shown. The HAM is simply executing the selected action twice. For fully working examples, see [examples folder](https://github.com/Juno-T/pyham/tree/master/examples).

### Designing HAM
A HAM is initialized as a python object while the machines are defined using python function definition. The use of python function definition expands the machines design limitation found in HAM's papers.
``` python
from pyham import HAM
from gym import spaces

# initializing HAM
myham = HAM(representation="onehot")

# defining choicepoint
my_choice_point = myham.choicepoint("my_binary_choicepoint", spaces.Discrete(2), discount=0.9)

# define and register machine
@myham.machine                                      # Use decorator .machine to register a machine
def infinite_loop_machine(ham: HAM):                # 1st argument is always ham. Machines can have additional arguments as well.
  while ham.is_alive:                               # Use while ham.is_alive instead of while True
      action = ham.CALL_choice(my_choice_point)     # Choice state
      ham.CALL_action(action)                       # Action state (executing primitive action)
      ham.CALL_action(action)
```

### Creating Wrapped HAM (induced MDP *H•M* in the original paper)
Here I utilize `create_concat_joint_state_env` which is a convenient wrapper function. Conversely, you can use lower level wrapper explained in [[RL library integration]] for more control.
```python
from pyham.wrappers.helpers import create_concat_joint_state_env

original_env = gym.make('CartPole-v1')

kwargs = {
  "initial_machine": infinite_loop_machine,
  "initial_args": [],
  "np_pad_config": {"constant_values": 0},
  "machine_stack_cap": 1,
  "eval": False,
  "will_render": False,
}

wrapped_env = create_concat_joint_state_env(ham, 
                                            original_env,
                                            **kwargs)
```
### Running wrapped env
Now you can use `wrapped_env` as if it's a `gym.Env`, i.e. through `reset`, `step` and `render` apis.  

A Few caveats:
* There may be multiple frames when calling `render` after each action execution.
* Apis for wrapped env of multi-choicepoints HAM will be according to [rllib's `MultiAgentEnv`](https://docs.ray.io/en/latest/rllib/rllib-env.html#multi-agent-and-hierarchical) instead of `gym.Env`.
