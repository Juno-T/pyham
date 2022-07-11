import unittest

from pyham.ham import HAM
from numpy.random import default_rng


class TestFunctionality(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    return super().setUpClass()

  def setUp(self) -> None:
    self.rng = default_rng(42)

    self.replay_buffer = []
    def env_exe(action):
      return "observation", 0.5, False, "info"
    def transition_handler(transition):
      self.replay_buffer.append(transition)
    discount = 1
    self.myham = HAM(env_exe, transition_handler, discount)
    return super().setUp()

  def test_initialize(self):
    @self.myham.learnable_choice_machine
    def m2_choice(ham: HAM, args):
      return 10

    @self.myham.action_machine
    def m3_action(ham: HAM, args):
      action = ham.CALL(m2_choice)
      return action

    @self.myham.functional_machine
    def loop_machine(ham: HAM, args):
      for i in range(10):
        ham.CALL(m3_action)

    self.assertTrue(self.myham.machine_count==3)

  def test_simple(self):
    @self.myham.learnable_choice_machine
    def m2_choice(ham: HAM, args):
      return "this is m2 choice"

    @self.myham.action_machine
    def m3_action(ham: HAM, args):
      action = ham.CALL(m2_choice)
      return action

    @self.myham.functional_machine
    def loop_machine(ham: HAM, args):
      for i in range(10):
        ham.CALL(m3_action)

    self.assertTrue(self.myham.machine_count==3)
    self.myham.episodic_reset("initial observation")
    self.assertTrue(self.myham._current_observation=="initial observation")
    self.myham.CALL(loop_machine)
    self.assertTrue(len(self.replay_buffer)==9)
    for transition in self.replay_buffer:
      self.assertTrue(transition.a_tm1=="this is m2 choice")

  
  def test_passing_machine_argument(self):
    @self.myham.learnable_choice_machine
    def m2_choice(ham: HAM, args):
      return "this is m2 choice"

    @self.myham.action_machine
    def m3_action(ham: HAM, args):
      if isinstance(args, dict) and "action" in args:
        if args["action"] is not None:
          action = args["action"]
      else:
        action = ham.CALL(m2_choice)
      return action

    @self.myham.functional_machine
    def loop_machine(ham: HAM, args):
      for i in range(10):
        ham.CALL(m3_action, {'action': i})

    self.assertTrue(self.myham.machine_count==3)
    self.myham.episodic_reset("initial observation")
    self.assertTrue(self.myham._current_observation=="initial observation")

    self.myham.CALL(loop_machine)
    self.assertTrue(len(self.replay_buffer)==0)
    obsv = self.myham.CALL(m3_action, {"action": "my action"})
    self.assertTrue(obsv=="observation")
  
  def test_rep_actions_transitions(self):
    @self.myham.learnable_choice_machine
    def repetition_choice(ham: HAM, args:int):
      return int(args)

    @self.myham.action_machine
    def action_machine(ham: HAM, args):
      if isinstance(args, dict) and "action" in args:
        if args["action"] is not None:
          action = args["action"]
      else:
        action = "whatever"
      return action

    @self.myham.functional_machine
    def loop_machine(ham: HAM, args):
      for i in range(10):
        rep = ham.CALL(repetition_choice, i)
        for _ in range(rep):
          ham.CALL(action_machine, {'action': "my action"})

    self.assertTrue(self.myham.machine_count==3)
    self.myham.episodic_reset("initial observation")
    self.assertTrue(self.myham._current_observation=="initial observation")

    self.myham.CALL(loop_machine)
    self.assertTrue(len(self.replay_buffer)==9)
    for i, transition in enumerate(self.replay_buffer):
      self.assertTrue(transition.a_tm1==i)
      self.assertTrue(transition.r_t == 0.5*i)

  def test_trivial_action_call(self):
    @self.myham.learnable_choice_machine
    def repetition_choice(ham: HAM, args:int):
      return int(args)

    @self.myham.functional_machine
    def loop_machine(ham: HAM, args):
      for i in range(10):
        rep = ham.CALL(repetition_choice, i)
        for _ in range(rep):
          ham.CALL_action("my action")

    self.assertTrue(self.myham.machine_count==2)
    self.myham.episodic_reset("initial observation")
    self.assertTrue(self.myham._current_observation=="initial observation")

    self.myham.CALL(loop_machine)
    self.assertTrue(len(self.replay_buffer)==9)
    for i, transition in enumerate(self.replay_buffer):
      self.assertTrue(transition.a_tm1==i)
      self.assertTrue(transition.r_t == 0.5*i)