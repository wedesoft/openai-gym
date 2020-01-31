import gym
env = gym.make('LunarLander-v2')
for e in range(3):
    done = False
    score = 0
    state = env.reset()
    for i in range(100):
        env.render()
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            break
env.close()
