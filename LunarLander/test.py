import gym
from gym import wrappers
import numpy as np
import os
import copy
import tensorflow as tf
from tfmodel import DQNAgent
import matplotlib as plt

import matplotlib.pyplot as plt

env = gym.make('LunarLander-v2')

#hyper parameters
adjustment_max_percentage = .1
generations = 100
trails_per_performance_evaluation = 5
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 64
n_episodes = 1000
target_model_update_interval = 500
output_dir = 'model_output/lunarlander/max_eval_24_10000_min-e-10_200target/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

agent = DQNAgent(state_size, action_size)

test_mode = True

if test_mode:
    agent.load(output_dir + 'weights_5150.hdf5')
    while True:
        state = env.reset() # reset state at start of each new episode of the game
        state = np.reshape(state, [1, state_size])
        cummulative_reward = 0
        for time in range(3000):  # time represents a frame of the game; goal is to keep pole upright as long as possible up to range, e.g., 500 or 5000 timesteps
            exploit = True
            env.render()
            action = agent.act(state, exploit) 
            next_state, reward, done, _ = env.step(action)# agent interacts with env, gets feedback; 4 state data points, e.g., pole angle, cart position        
            cummulative_reward += reward
            next_state = np.reshape(next_state, [1, state_size])       
            state = next_state # set "current state" for upcoming iteration to the current next state 
            if done: # episode ends if agent drops pole or we reach timestep 5000
                break # exit loop


maximum_evaluation = -1000
done = False
scores = []
episodes_log = []
plt.ion()
plt.style.use('default')
plt.ylim(-300,300)
plt.xlim(0,n_episodes)
for e in range(n_episodes): # iterate over new episodes of the game
    state = env.reset() # reset state at start of each new episode of the game
    state = np.reshape(state, [1, state_size])
    cummulative_reward = 0

    #every 50th episode measure performance, by running 3 trials and averaging result.
    if e % 50 == 0:
        total_reward = 0
        for i in range(trails_per_performance_evaluation):
            state = env.reset() # reset state at start of each new episode of the game
            state = np.reshape(state, [1, state_size])
            cummulative_reward = 0
            for time in range(1000): 
                exploit = True
                action = agent.act(state, exploit) 
                next_state, reward, done, _ = env.step(action)    
                cummulative_reward += reward
                next_state = np.reshape(next_state, [1, state_size])       
                env.render()
                state = next_state # set "current state" for upcoming iteration to the current next state 
                if done: # episode ends if agent drops pole or we reach timestep 1000
                    break # exit loop
            total_reward += cummulative_reward
        average_reward = total_reward / trails_per_performance_evaluation
        scores.append(average_reward)
        episodes_log.append(e)
        plt.style.use('default')
        plt.ylim(-300,300)
        plt.xlim(0,n_episodes)
        plt.plot(episodes_log, scores)
        plt.show()
        plt.pause(0.001)
        if average_reward > maximum_evaluation:
            maximum_evaluation = average_reward
            agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")
        plt.savefig(output_dir + 'graph.png')

        if e % target_model_update_interval == 0 and agent.epsilon > 0.25:
            agent.update_target_model()
        elif average_reward > 200:
            agent.update_target_model()



        

            
    for time in range(1000):  # time represents a frame of the game; goal is to keep pole upright as long as possible up to range, e.g., 500 or 5000 timesteps
        exploit = False
        action = agent.act(state, exploit) # action is either 0 or 1 (move cart left or right); decide on one or other here
        next_state, reward, done, _ = env.step(action) # agent interacts with env, gets feedback; 4 state data points, e.g., pole angle, cart position        
        cummulative_reward += reward
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done) # remember the previous timestep's state, actions, reward, etc.        
        state = next_state # set "current state" for upcoming iteration to the current next state 

        if done: # episode ends if agent drops pole or we reach timestep 5000
            print("episode: {}/{}, score: {:02f}, time: {}, e: {:.2}" # print the episode's score and agent's epsilon
                  .format(e, n_episodes, cummulative_reward, time, agent.epsilon))
            break # exit loop
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size) # train the agent by replaying the experiences of the episode



        


