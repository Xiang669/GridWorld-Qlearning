import numpy as np
import matplotlib.pyplot as plt

import environment

from agent import QlearningAgent

def train():
    env = environment.GridWorld()
    agent = QlearningAgent(env)

    num_episodes = 10000
    returns = []
    for episode in range(num_episodes):
        state = env.reset()
        state_index = agent.state_to_index(state)
        total_reward = 0

        while not env.done:
            action = agent.choose_action_with_exploration(state_index)
            next_state, reward, env.done = env.step(action)
            next_state_index = agent.state_to_index(next_state)

            agent.update_q_value(state_index, action, reward, next_state_index)

            total_reward += reward
            state_index = next_state_index

        returns.append(total_reward)

    #print("Q-Table:")
    #print(agent.q_table)
    return agent.get_policy(),returns
        
        

# turn the action index to an arrow for visualization
ARROWS={0: "↑", 1: "↓", 2: "←", 3: "→"}

def main():
    policy, returns = train()
    print("Learned Policy:")
    for i in range(environment.GridWorld.ROWS):
        for j in range(environment.GridWorld.COLS):
            state_index = i * environment.GridWorld.COLS + j
            action = policy[state_index]
            if (i, j) == environment.GridWorld.GOAL:
                print("G", end=" ")
            else:
                print(ARROWS[action], end=" ")
        print()
    plot_returns(returns, window=100)

def plot_returns(returns, window=100):
    returns = np.asarray(returns)
    episodes = np.arange(1, len(returns) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        episodes, returns, color="blue",
        alpha=0.5, linewidth=0.5, label="total rewards"
    )

    window = min(window, len(returns))
    moving_average = np.convolve(
        returns, np.ones(window) / window, mode="valid"
    )
    ax.plot(
        episodes[window - 1:], moving_average,color="orange",
        linewidth=2, label=f"{window}-episode moving average"
    )

    ax.axhline(
        y=4, color="red", linestyle="--",
        label="Optimal return = 4"
    )

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Total reward (undiscounted)")
    ax.set_title("Q-learning training returns")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("training_returns.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()
    