import time
import gym
env = gym.make('LunarLander-v2')
env.reset()
done = False
while not done:
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    time.sleep(0.04)
env.close()
