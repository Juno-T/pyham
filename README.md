# Python implementation of Hierarchies of Abstract Machine (HAM)

Hierarchies of Abstract Machine, also known as HAM, is a framework for hierarchical reinforcement learning (HRL) which was first introduced in [this paper](https://proceedings.neurips.cc/paper/1997/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)(Parr & Russell, 1997).

:construction: The project is still in progress.

## installation
```
git clone https://github.com/Juno-T/pyham.git
export PYTHONPATH=$PWD:$PYTHONPATH
cd pyham
pip install -r requirements.txt             // requirements for core ham
pip install -r examples/requirements.txt    // requirements for running examples

// Try running examples
wandb login
python3 examples/sb3_cartpole/train_ppo.py --machine trivial      // can replace trivial with balance-recover
python3 examples/sb3_taxi/train_dqn.py --machine trivial          // can replace trivial with get-put
```
