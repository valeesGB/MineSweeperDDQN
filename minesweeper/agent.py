import random
import numpy as np
from DDQN import *
from ReplayBuffer import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


LR = 0.01               # learning rate
LEARN_MIN = 0.001
LEARN_DECAY = 0.99975

eps = 0.9 # Epsilon for epsilon-greedy action selection
EPSILON_DECAY = 0.99975
EPSILON_MIN = 0.01

MEM_SIZE = 50000  # replay buffer size
MEM_SIZE_MIN = 1000  # minimum size of replay buffer before training starts
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.1            # discount factor
UPDATE_EVERY = 20       # how often to update the network (When Q target is present)


class MinesweeperAgent:
    def __init__(self, state_size, action_size, seed = 42, adv_type = 'avg'):
        
        self.epsilon = eps
        self.learn_rate = LR
        self.gamma = GAMMA  
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.qnetwork_local = QNetwork1(state_size, action_size, seed, adv_type).to(device)
        self.qnetwork_target = QNetwork1(state_size, action_size, seed, adv_type).to(device)
        self.optimizer = torch.optim.AdamW(self.qnetwork_local.parameters(), lr=LR)
        
        self.memory = ReplayBuffer(action_size, MEM_SIZE, BATCH_SIZE, seed)
        
        self.t_step = 0  # Initialize time step (for updating every UPDATE_EVERY steps)
        
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and learn every UPDATE_EVERY time steps."""
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        
        if len(self.memory) >= MEM_SIZE_MIN and self.t_step == 0:
            experiences = self.memory.sample()
            self.learn(experiences)
             # Update the target network
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
           
            
    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        """self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))"""
            
            
        #self.qnetwork_local.train()

        # =============================================================================
        # 1) Maschero i Q-values delle posizioni già scoperte
        #    (assumo che `state` sia un array 1D di lunghezza action_size,
        #     con -1 = non rivelata, >=0 = rivelata.)
        # =============================================================================
        # portiamo a CPU per ricavare la maschera
        state_flat = state.flatten().cpu().numpy()
        solved_mask = (state_flat != -1)             # True = già rivelata

        # calcolo il valore minimo tra tutti i Q
        q = action_values                       # shape [1, action_size]
        min_q = q.min(dim=1, keepdim=True)[0]   # shape [1,1]

        # applico la maschera: le caselle rivelate prendono il min_q
        # costruisco una mask tensor [1,action_size]
        mask = torch.tensor(solved_mask,
                            dtype=torch.bool,
                            device=device).unsqueeze(0)
        q = q.masked_scatter(mask, min_q.expand_as(q)[mask])

        # =============================================================================
        # 2) ε‐greedy solo sulle caselle non rivelate
        # =============================================================================
        if random.random() > self.epsilon:
            # pick massima Q fra quelle “sane”
            action = q.argmax(dim=1).item()
        else:
            # esploro scegliendo un indice tra quelli non rivelati
            candidates = np.where(state_flat == -1)[0]
            action = np.random.choice(candidates)

        return action
        
    def learn(self, experiences):
        
        """Update value parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences
        
        self.qnetwork_local.train()
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones)) #non sono sicuro di (1 - dones)
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        
        ''' Gradiant Clipping '''
        """ +T TRUNCATION PRESENT """
        for param in self.qnetwork_local.parameters():
            if param.grad != None:
              param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
'''        
        #decay epsilon
        self.epsilon = max(EPSILON_MIN, self.epsilon*EPSILON_DECAY)'''
