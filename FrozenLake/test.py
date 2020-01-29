import gym
import matplotlib.pyplot as plt
from agents.q_learning_agent import Agent

n_tests = 10
n_episodes = 100

env = gym.make("FrozenLake-v0", is_slippery=False)
env.reset()        

agent = Agent()
agent.initStateActionSpace(16, 4)

print(env.action_space)
print(env.observation_space)
rewards = []
test_indicies = []
total_reward = 0

state_action_pairs = []
rewards = []

for test in range(0, n_tests):
    for episode in range(0 , n_episodes):
        state = 0
        state_action_pairs.clear()
        env.reset()
        for step_number in range(0 , 20):
            action = agent.takeAction(state)
            
            #add the new state action pair into our list for this episode
            state_action_pairs.append([state, action])

            state, reward, done, info = env.step(action)
            #env.render()

            if done or step_number == 19:
                if reward == 0:
                    agent_reward = -1
                else:
                    agent_reward = 1
                agent.train(state_action_pairs, agent_reward)
                total_reward += reward
                break
            
    rewards.append(total_reward)
    test_indicies.append(test)
    total_reward = 0


plt.style.use('default')
plt.ylim(0,n_episodes*1.1)
plt.xlim(0,n_tests)
plt.plot(test_indicies, rewards)
plt.ylabel('Successful Trials in Episode')
plt.xlabel('Episode')
plt.title('Frozen Lake (Deterministic): Q Learning Agent Performance')
plt.show()

