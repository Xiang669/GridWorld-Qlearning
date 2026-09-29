class GridWorld:

    ROWS = 6
    COLS = 5

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

    def __init__(self):
        self.start_state = self.START
        self.goal_state = self.GOAL
        self.state=self.start_state
        self.step_reward = self.STEP_REWARD
        self.goal_reward = self.GOAL_REWARD

    def reset(self):
        self.state = self.START
        return self.state

    def step(self, action):
        if action not in self.ACTIONS:
            raise ValueError("Invalid action")

        # Get the current position
        row, col = self.state

        # Calculate new position based on action
        delta_row, delta_col = self.ACTIONS[action]
        new_row = row + delta_row
        new_col = col + delta_col

        # Check for boundaries
        if 0 <= new_row < self.ROWS and 0 <= new_col < self.COLS:
            self.state = (new_row, new_col)
        else:
            # If out of bounds, stay in the same state
            self.state = (row, col)

        # Check if goal is reached
        if self.state == self.GOAL:
            reward = self.goal_reward
            done = True
        else:
            reward = self.step_reward
            done = False

        return self.state, reward, done
