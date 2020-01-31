import time
import gym
env = gym.make('LunarLander-v2')
env.reset()
print(env.action_space)
print(env.observation_space)
print(env.observation_space.low)
print(env.observation_space.high)
done = False
while not done:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(observation, reward, done, info)
    time.sleep(0.04)
env.close()
