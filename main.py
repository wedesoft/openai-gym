# https://github.com/shivaverma/OpenAIGym/blob/master/lunar-lander/discrete/lunar_lander.py
import collections
import numpy as np
import keras
import gym


class DQN:
    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.memory = collections.deque(maxlen=100000)

    def build_model(self):
        # Model taking state as input and outputting the expected reward for each action.
        model = keras.Sequential()
        model.add(keras.layers.Dense(150, input_dim=self.state_space.shape[0], activation=keras.activations.relu))
        model.add(keras.layers.Dense(120, activation=keras.activations.relu))
        model.add(keras.layers.Dense(self.action_space.n, activation=keras.activations.linear))
        model.compile(loss='mse', optimizer=keras.optimizers.adam(lr=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()
        values = self.model.predict(state.reshape(1, 8))[0]
        return np.argmax(values)

    def remember(self, *args):
        self.memory.append(args)


env = gym.make('LunarLander-v2')
agent = DQN(env.action_space, env.observation_space)
for e in range(3):
    done = False
    score = 0
    state = env.reset()
    for i in range(100):
        env.render()
        old_state = state
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        score += reward
        agent.remember(old_state, action, reward, state, done)
        if done:
            break
env.close()
