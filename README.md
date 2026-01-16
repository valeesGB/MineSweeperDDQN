# MineSweeperDDQN

A Deep Reinforcement Learning project that trains an AI agent to play **Minesweeper** using a **Double Deep Q-Network (DDQN)**.

## 📌 Project Overview

This repository contains an implementation of a reinforcement learning agent designed to solve the classic game of Minesweeper. Unlike standard Q-Learning, this project utilizes **Double DQN**, which helps reduce the overestimation of action values often found in vanilla DQN, leading to more stable and efficient training.

### Key Features
*   **Minesweeper Environment**: A custom or simulated environment where the agent interacts with the grid.
*   **Double Deep Q-Network (DDQN)**: Uses two neural networks (Online Network and Target Network) to decouple action selection from target value estimation.
*   **Experience Replay**: Stores past transitions to break correlation between consecutive samples during training.
*   **Convolutional Neural Network (CNN)**: Processes the grid state as an image-like input to capture spatial dependencies between cells.

## 🛠️ Requirements

The project relies on **Python 3.x** and the following libraries.

#### Core Libraries
*   **Jupyter**: To run the `.ipynb` notebooks.
*   **Gym** (OpenAI Gym): For the Minesweeper game environment.
*   **NumPy**: For grid manipulation and state processing.
*   **Matplotlib**: For plotting training performance (win rates/loss).

#### Deep Learning Framework
*   **PyTorch**: For the Double Deep Q-Network (DDQN).

#### Installation
You can install all necessary dependencies with the following command:

```bash
pip install numpy matplotlib gym torch jupyter
```

## 🚀 Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/valeesGB/MineSweeperDDQN.git
    cd MineSweeperDDQN
    ```

2.  **Navigate to the project folder:**
    ```bash
    cd minesweeper
    ```

3.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

4.  **Run the Training:**
    Open the main notebook file (e.g., `MineSweeper_DDQN.ipynb` or similar) and run the cells to start training the agent. The notebook typically includes:
    *   Environment initialization.
    *   Network architecture definition.
    *   Training loop (episodes, exploration vs. exploitation).
    *   Visualization of win rates and loss.

## 🧠 How It Works

### The State
The Minesweeper grid is represented as a matrix where values indicate:
*   **-1**: Unexplored cell
*   **0-8**: Number of adjacent mines (revealed)
*   **-2** (or similar): Flagged cell / Boundary

### The Action
The agent chooses a coordinate $(x, y)$ on the grid to reveal.

### The Reward
*   **Positive Reward**: For revealing a safe cell (often scaled by the number of adjacent mines revealed).
*   **Negative Reward**: For hitting a mine (Game Over).
*   **Win Reward**: A large bonus for clearing the board.

### DDQN Algorithm
1.  **Select Action**: Uses an $\epsilon$-greedy policy (explores randomly at first, then exploits the model).
2.  **Step**: The agent acts on the environment and receives a `next_state`, `reward`, and `done` flag.
3.  **Store**: The transition is saved in the **Replay Buffer**.
4.  **Train**: A batch of random transitions is sampled from the buffer. The model minimizes the loss between the predicted Q-value and the target Q-value calculated using the Target Network.

## 📈 Results

*   *Training Progress*: (Add plots of Win Rate vs. Episodes here if available)
*   *Performance*: The agent typically learns to solve beginner grids quickly and improves on larger grids over thousands of episodes.

## 🤝 Contributing

Contributions are welcome! If you have ideas for improving the model architecture, optimizing the reward function, or fixing bugs, feel free to open an issue or submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Author**: [valeesGB](https://github.com/valeesGB)
