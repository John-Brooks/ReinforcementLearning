import gym
import matplotlib.pyplot as plt

n_tests = 100
n_episodes = 1000
 
env = gym.make("FrozenLake-v0", is_slippery=False)
env.reset()                    
print(env.action_space)
print(env.observation_space)
rewards = []
test_indicies = []
total_reward = 0

for test in range(0, n_tests):
    for episode in range(0 , n_episodes):
        env.reset()
        for step_number in range(0 , 20):
            action = env.action_space.sample()
            new_state, reward, done, info = env.step(action)
            if done:
                total_reward += reward
                break
            #env.render()
    rewards.append(total_reward)
    test_indicies.append(test)
    total_reward = 0


plt.style.use('default')
plt.ylim(0,n_episodes)
plt.xlim(0,n_tests)
plt.plot(test_indicies, rewards)
plt.ylabel('Successful Trials in Episode')
plt.xlabel('Episode')
plt.title('Frozen Lake (Deterministic): Random Agent Performance')
plt.show()

