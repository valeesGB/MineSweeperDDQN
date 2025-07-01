import numpy as np
import gym 
from gym import spaces

class MinesweeperEnv(gym.Env):
    def __init__(self, grid_size=9, num_mines=10):
        super(MinesweeperEnv, self).__init__()
        
        self.grid_size = grid_size
        self.num_mines = num_mines
        
        self.action_space = spaces.Discrete(grid_size * grid_size)
        
        #-1: hidden cell, 0+: revealed numbers
        self.observation_space = spaces.Box(low=-1, high=8, shape=(grid_size, grid_size), dtype=np.int8)
        
        self.board = None 
        self.state = None 
        self.mines = None
        self.done = False
        self.is_first_step = True
        
    def reset(self):
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.state = np.full((self.grid_size, self.grid_size), -1, dtype=np.int32)
        
        self.done = False
        
        # Place mines randomly
        self.mines = set()
        while len(self.mines) < self.num_mines:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            if (x, y) not in self.mines:
                self.mines.add((x, y))        
        
        for x, y in self.mines:
            for dx in range (-1, 2):
                for dy in range(-1, 2):
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and (nx, ny) not in self.mines:
                        self.board[nx, ny] += 1
        return self.state.flatten()
        
    def getExperiences(self, action):
        """
        Processes the agent's action and returns the resulting experience tuple.
        Args:
            action (int): The action to take, represented as a flattened index of the grid.
        Returns:
            tuple: A tuple containing:
                - next_state (np.ndarray): The flattened state of the environment after the action.
                - reward (int): The reward received for the action.
                - done (bool): Whether the episode has ended.
                - info (dict): Additional information (empty by default).
        Raises:
            RuntimeError: If the environment has already reached a terminal state and needs to be reset.
        Notes:
            - If the selected cell is already revealed, a penalty is applied.
            - If a mine is hit, the episode ends with a large negative reward.
            - If an empty cell is revealed, adjacent cells may also be revealed.
            - If all non-mine cells are revealed, the episode ends with a large positive reward.
        """
        if self.done:
            raise RuntimeError("Environment is done. Please reset it.")
            
        row, col = divmod(action, self.grid_size)
        
        if (row, col) in self.mines:
            self.state[row, col] = -1 # Mine hit
            self.done = True
            reward = -1
        else:
            if self.is_first_step:
                reward = 0
                self.is_first_step = False
            else:
                reward = -0.3 if self.guess_neighbors(row, col) else 0.3
            
            self.state[row, col] = self.board[row, col]
            
            
            if self.board[row, col] != -1:
                # Reveal adjacent cells if the cell is empty
                self._reveal_adjacent_cells(row, col)
                
            if self._check_win():
                self.done = True
                reward = 1
                print("You win!")
                
        return self.state.flatten(), reward, self.done #return state in realtà ritorna next_state
        
    def render(self, mode='human'):
        render_board = ''
        for row in self.state:
            render_board += ' '.join(str(cell) if cell>= 0 else '□' for cell in row) + '\n'
        print(render_board)
        
    def guess_neighbors(self, row, col):
        """Get the neighboring cells of a given cell."""
        neighbors = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx = row + dx
                ny = col + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    neighbors.append((nx, ny))

        return bool(np.sum([self.state[values] for values in neighbors]) < 0)
    
    def _reveal_adjacent_cells(self, row, col):
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx = row + dx
                ny = col + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.state[nx, ny] == -1 and (nx, ny) not in self.mines: # Only reveal hidden cells
                        self.state[nx, ny] = self.board[nx, ny]
                        if self.board[nx, ny] == 0:
                            self._reveal_adjacent_cells(nx, ny)
                            
    def _check_win(self):
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in self.mines and self.state[x, y] == -1:
                    return False
        return True