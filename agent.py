import numpy as np

class QlearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

        # Initialize Q-table
        self.q_table = np.zeros((env.ROWS * env.COLS, len(env.ACTIONS)))

    def choose_action_with_exploration(self, state_index):
        if np.random.rand() < self.epsilon:
            # Explore: choose a random action
            return np.random.choice(list(self.env.ACTIONS.keys()))
        else:
            # Exploit: choose the best action from Q-table
            return self.choose_action(state_index)

    def choose_action(self, state_index):
        # Choose the best action from Q-table
        max=np.max(self.q_table[state_index])
        best_actions = np.where(self.q_table[state_index] == max)[0]
        return np.random.choice(best_actions)

    def update_q_value(self, state_index, action, reward, next_state_index):
        best_next_action = self.choose_action(next_state_index)
        td_target = reward + self.gamma * self.q_table[next_state_index][best_next_action]
        td_delta = td_target - self.q_table[state_index][action]
        self.q_table[state_index][action] += self.alpha * td_delta

    def get_policy(self):
        policy = {}
        for state_index in range(self.q_table.shape[0]):
            best_action = self.choose_action(state_index)
            policy[state_index] = best_action
        return policy

    def state_to_index(self, state):
        row, col = state
        return row * self.env.COLS + col
