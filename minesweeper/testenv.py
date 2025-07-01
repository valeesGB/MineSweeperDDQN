from environment import MinesweeperEnv

# Create the environment
env = MinesweeperEnv(grid_size=9, num_mines=10)

# Reset the environment
state = env.reset()
done = False

# Manual play (for debugging)
while not done:
    env.render()  # Display the board
    action = int(input("Enter action (cell index 0-80): "))
    state, reward, done, _ = env.getExperiences(action)
    print(f"cre stu done: {done}")
    print(f"Reward: {reward}")

if reward == 100:
    print("You won!")
else:
    print("Game Over!")
    
    
    
    
if __name__ == "__main__":
    # Run the manual play
    while True:
        try:
            env.render()  # Display the board
            action = int(input("Enter action (cell index 0-80): "))
            state, reward, done, _ = env.getExperiences(action)
            print(f"State: {state}")
            print(f"Reward: {reward}")
            if done:
                if reward == 100:
                    print("You won!")
                else:
                    print("Game Over!")
                break
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 80.")
        except RuntimeError as e:
            print(e)
            break