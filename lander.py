# https://github.com/shivaverma/OpenAIGym/
import os
import pickle
import random
import collections
import ruamel.yaml as yaml
import warnings
import numpy as np
import keras
import gym
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)


class Agent:
    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.epsilon_min = 0.05
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory = collections.deque(maxlen=1000000)
        self.weights_file = 'lander.h5'
        self.conf_file = 'config.yml'
        self.memory_file = 'memory.pickle'
        self.load_config()
        self.load_memory()
        self.load_memory()
        self.model = self.build_model()

    def load_config(self):
        if not os.path.exists(self.conf_file):
            return
        with open(self.conf_file) as f:
            config = yaml.load(f)
            self.epsilon = config['epsilon']

    def load_memory(self):
        if not os.path.exists(self.memory_file):
            return
        with open(self.memory_file, 'rb') as f:
            self.memory = pickle.load(f)

    def build_model(self):
        # Model taking state as input and outputting the expected reward for each action.
        model = keras.Sequential()
        model.add(keras.layers.Dense(150, input_dim=self.state_space.shape[0], activation=keras.activations.relu))
        model.add(keras.layers.Dense(120, activation=keras.activations.relu))
        model.add(keras.layers.Dense(self.action_space.n, activation=keras.activations.linear))
        model.compile(loss='mse', optimizer=keras.optimizers.adam(lr=self.learning_rate))
        if os.path.exists(self.weights_file):
            model.load_weights(self.weights_file)
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()
        values = self.model.predict(state.reshape(1, 8))[0]
        return np.argmax(values)

    def remember(self, *args):
        self.memory.append(args)

    def replay(self, *args):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        # Get estimated maximum reward for next state for each sample.
        next_reward = np.amax(self.model.predict_on_batch(next_states), axis=-1)
        # Compute reward for current action for each sample.
        targets = rewards + self.gamma * next_reward * (1 - dones)
        # Get current estimated rewards for each action from neural network.
        current = self.model.predict_on_batch(states)
        # Update part of the table with new rewards.
        current[list(range(self.batch_size)), actions] = targets
        # Perform training step with updated table.
        self.model.fit(states, current, epochs=1, verbose=0)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self):
        self.model.save_weights(self.weights_file)
        with open(self.conf_file, 'w') as f:
            yaml.dump({'epsilon': self.epsilon}, f)
        with open(self.memory_file, 'wb') as f:
            pickle.dump(self.memory, f)


env = gym.make('LunarLander-v2')
agent = Agent(env.action_space, env.observation_space)
episodes = 100
max_steps = 3000 # need to fly long enough for fuel penalty to overcome penalty for crashing
for e in range(episodes):
    print('Episode =', e)
    done = False
    score = 0
    state = env.reset()
    for i in range(max_steps):
        env.render()
        old_state = state
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        score += reward
        agent.remember(old_state, action, reward, state, done)
        agent.replay()
        if done:
            break
    print('score =', score, 'last reward =', reward)
env.close()
agent.save()
