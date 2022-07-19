import gym
from stable_baselines3 import PPO

from pyham.ham import create_concat_joint_state_wrapped_env
from pyham.examples.utils import render_frames

from machines import create_trivial_cartpole_ham


def main():
  config = {
    "internal_discount": 0.95,
    "machine_stack_cap": 1,
    "constant_value": 0
  }
  model_path="ppo_trivial_cartpole"
  video_path="ppo_trivial_cartpole_render.mp4"

  original_env = gym.make("CartPole-v1", new_step_api=False)
  ham, choice_space, initial_machine, initial_args = create_trivial_cartpole_ham(config["internal_discount"])
  
  wrapped_env = create_concat_joint_state_wrapped_env(ham, 
                            original_env, 
                            choice_space, 
                            initial_machine=initial_machine,
                            initial_args=initial_args,
                            np_pad_config = {"constant_values": config["constant_value"]},
                            machine_stack_cap=config["machine_stack_cap"],
                            will_render=False)
                        
  model = PPO("MlpPolicy", wrapped_env, verbose=1)
  model.learn(total_timesteps=25000)
  model.save(model_path)
  wrapped_env.close()

  wrapped_env.set_render_mode(True)
  model = PPO.load(model_path)
  obs = wrapped_env.reset()
  frames = wrapped_env.render()
  done=False
  while not done:
      action, _states = model.predict(obs)
      obs, reward, done, info = wrapped_env.step(action)
      frames+=wrapped_env.render()
  render_frames(frames, video_path)
  wrapped_env.set_render_mode(False)

if __name__=="__main__":
  main()