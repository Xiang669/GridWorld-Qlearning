import numpy as np

class QlearningAgent:

    #initialize the Q-learning agent with environment,
    #learning rate, discount factor, and exploration rate
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        # initialize Q-table
        self.q_table = np.zeros((env.ROWS * env.COLS, len(env.ACTIONS)))

    #epsilon-greedy action selection with exploration
    def choose_action_with_exploration(self, state_index):
        if np.random.rand() < self.epsilon:
            # explore: choose a random action
            self.epsilon=max(0.01, self.epsilon * 0.995)  # decay epsilon
            return np.random.choice(list(self.env.ACTIONS.keys()))
        else:
            # exploit: choose the best action from Q-table
            return self.choose_action(state_index)

    #choose the best action from Q-table
    def choose_action(self, state_index):
        #choose the best action from Q-table
        max=np.max(self.q_table[state_index])
        best_actions = np.where(self.q_table[state_index] == max)[0]
        return np.random.choice(best_actions)

    #update the Q-value for the given state and action based on
    #the received reward and next state
    def update_q_value(self, state_index, action, reward, next_state_index):
        best_next_action = self.choose_action(next_state_index)
        td_target = reward + self.gamma * self.q_table[next_state_index][best_next_action]
        td_delta = td_target - self.q_table[state_index][action]
        self.q_table[state_index][action] += self.alpha * td_delta

    #get the learned policy by selecting the best action 
    #for each state based on the Q-table
    def get_policy(self):
        policy = {}
        for state_index in range(self.q_table.shape[0]):
            best_action = self.choose_action(state_index)
            policy[state_index] = best_action
        return policy

    #convert a state (row, col) to a unique index for the Q-table
    def state_to_index(self, state):
        row, col = state
        return row * self.env.COLS + col
