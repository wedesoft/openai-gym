import gym
env = gym.make('LunarLander-v2')
for e in range(10):
    done = False
    state = env.reset()
    while not done:
        env.render()
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
env.close()
