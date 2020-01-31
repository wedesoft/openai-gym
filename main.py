import time
import gym
env = gym.make('LunarLander-v2')
state = env.reset()
print(env.action_space)
print(env.observation_space)
print(env.observation_space.low)
print(env.observation_space.high)
done = False
while not done:
    env.render()
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    print(state, reward, done)
    time.sleep(0.04)
env.close()
