from typing import NamedTuple, Any
import threading

class JointState(NamedTuple):
  s: Any # Environment pure state
  m: Any # HAM's machine stack
  tau: int # number of timesteps since the previous choice point

class Transition(NamedTuple):
  s_tm1: JointState
  a_tm1: Any # choice
  r_t: float # cumulative reward between choice point
  s_t: JointState
  done: int # 1 if done, else 0

class AlternateLock():
    def __init__(self, initial_thread):
        self.allow = initial_thread
        self.cond = threading.Condition()
        
    def acquire_for(self, thread):
        self.cond.acquire()
        while self.allow!=thread:
            self.cond.wait()
    
    def release_to(self, thread):
        self.allow=thread
        self.cond.notifyAll()
        self.cond.release()
