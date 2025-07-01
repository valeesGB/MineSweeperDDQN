# train.py

from environment import MinesweeperEnv
from agent import MinesweeperAgent

if __name__ == "__main__":
    env = MinesweeperEnv()
    agent  = MinesweeperAgent(env.grid_size**2, env.grid_size**2)
    
    
    scores = []  # list containing scores from each episode
    n_episodes = 150  # number of episodes to run
    state = env.reset().flatten()
    i_episode = 0
    i_moves = 0
    total_reward = 0

    while (i_episode <= n_episodes):
        done = False

        action = agent.act(state)
        next_raw, reward, done = env.getExperiences(action)
        next_state = next_raw.flatten()

        agent.step(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward
        i_moves += 1
        if done:
            i_episode += 1
            print(f"Episode {i_episode} finished in {i_moves} with score: {total_reward}")
            scores.append(total_reward)
            state = env.reset().flatten()
            i_moves = 0
            total_reward = 0
