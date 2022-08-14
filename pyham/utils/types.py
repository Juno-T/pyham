from typing import NamedTuple, Any, NewType

class JointState(NamedTuple):
  s: Any # Environment pure state
  m: Any # HAM's machine stack
  tau: int # number of timesteps since the previous choice point

class Transition(NamedTuple):
  s_tm1: JointState
  a_tm1: Any # choice
  r_t: float # cumulative reward between choice point
  s_t: JointState
  done: int # 1 if don

from ..wrappers.single_choice import SingleChoiceTypeEnv

InducedMDP = NewType("InducedMDP", SingleChoiceTypeEnv)