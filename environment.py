import numpy as np
class GridWorld:

    ROWS = 5
    COLS = 4

    START = (0,0)
    GOAL = (4,3)

    ACTIONS = {
            0:(-1,0), #UP
            1:(1,0),  #DOWN
            2:(0,-1), #LEFT
            3:(0,1),  #RIGHT
        }

    STEP_REWARD = -1
    GOAL_REWARD = 10

    #initialize the environment
    def __init__(self):
        self.start_state = self.START
        self.goal_state = self.GOAL
        self.state=self.start_state
        self.step_reward = self.STEP_REWARD
        self.goal_reward = self.GOAL_REWARD
        self.done = False  # initialize done flag

    #reset the environment to the start state
    def reset(self):
        self.state = self.START
        self.done = False
        return self.state

    #reset the environment to a random state
    def random_reset(self):
        self.state = (np.random.randint(0, self.ROWS), np.random.randint(0, self.COLS))
        self.done = False
        return self.state

    #step function to take an action and return
    #the next state, reward, and done flag
    def step(self, action):
        if action not in self.ACTIONS:
            raise ValueError("Invalid action")

        # get the current position
        row, col = self.state

        # calculate new position based on action
        delta_row, delta_col = self.ACTIONS[action]
        new_row = row + delta_row
        new_col = col + delta_col

        # check for boundaries
        if 0 <= new_row < self.ROWS and 0 <= new_col < self.COLS:
            self.state = (new_row, new_col)
        else:
            # if out of bounds, stay in the same state
            self.state = (row, col)

        # check if goal is reached
        if self.state == self.GOAL:
            reward = self.goal_reward
            self.done = True
        else:
            reward = self.step_reward
            self.done = False

        return self.state, reward, self.done
