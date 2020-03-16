import tensorflow as tf
import numpy as np
from tensorflow import feature_column
from collections import deque
import random

class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=1000000)

        #how much to discount future rewards
        self.gamma = 0.99

        #exploration rate of the agent (1 = 100% exploration vs. exploitation)
        self.epsilon = 1.0
        self.epsilon_decay = 0.996
        self.epsilon_min = 0.01

        self.learning_rate = 0.001

        self.model = self._build_model()
        self.target_model = self.model

    def _build_model(self):

        # numeric cols
        network = tf.keras.models.Sequential([ 
        tf.keras.layers.Dense(24, input_dim = self.state_size, activation='relu'),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(self.action_size, activation='linear')
        ])

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        network.compile(optimizer=self.optimizer,
                loss='mse',
                )

        return network

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, exploit):
        
        #explore or exploit
        if (np.random.rand() <= self.epsilon) and not exploit:
            #act randomly
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            if done:
                target = reward
            else:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def update_target_model(self):
        self.target_model = self.model
   
