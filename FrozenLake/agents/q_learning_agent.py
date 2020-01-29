import random

class Agent:
    learning_rate = 0.1
    epsilon_greedy = 1.0 
    epsilon_decay = 0.01
    discount_factor = 0.8
    q_table = []

    def initStateActionSpace(self, state_size, action_size):
        self.q_table.clear()
        for _ in range(0, state_size):
            action = []
            for _ in range(0, action_size):
                action.append(0)
            self.q_table.append(action)

    def train(self, state_action_pairs, reward):
        discounted_reward = reward * self.learning_rate
        state_action_pairs.reverse()
        for pair in state_action_pairs:
            self.q_table[pair[0]][pair[1]] += discounted_reward
            discounted_reward = discounted_reward * self.discount_factor
        self.epsilon_greedy = self.epsilon_greedy - self.epsilon_decay
        if self.epsilon_greedy < 0:
            self.epsilon_greedy = 0

    def takeRandomAction(self):
        return random.randrange(0, 3)

    def takeGreedyAction(self, state):
        best_action = max(self.q_table[state])
        action_index = 0
        for action in self.q_table[state]:
            if action == best_action:
                return action_index 
            action_index += 1
        return 0

    def takeAction(self, state):
        if random.random() > self.epsilon_greedy:
            return self.takeGreedyAction(state)
        else:
            return self.takeRandomAction()

        

    

                