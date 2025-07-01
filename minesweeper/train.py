# train.py

from environment import MinesweeperEnv
from agent import MinesweeperAgent
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    env = MinesweeperEnv()
    agent  = MinesweeperAgent(env.grid_size**2, env.grid_size**2)
    
    writer = SummaryWriter()

    scores = []  # list containing scores from each episode
    n_episodes = 150000  # number of episodes to run
    state = env.reset().flatten()
    i_episode = 0
    i_moves = 0
    total_reward = 0
    win_count = 0  # count of wins in the last 500 episodes

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
            win_count +=1 if reward == 1 else win_count
            writer.add_scalar('Reward', total_reward, i_episode)
            writer.add_scalar('Number of moves', i_moves, i_episode)
            if i_episode+1 % 500 == 0:
                writer.add_scalar('Win_rate over 500 episodes', win_count/500, i_episode)
                win_count = 0
            i_episode += 1
            state = env.reset().flatten()
            i_moves = 0
            total_reward = 0

