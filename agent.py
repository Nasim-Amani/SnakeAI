import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point  # Import the game environment
from model import Linear_QNet , QTrainer
from plot import plot

# Constants for training
MAX_MEMORY = 100_000  # Maximum memory storage for experience replay
BATCH_SIZE = 1000  # Number of experiences used for training at once
LR = 0.001  # Learning rate for training the AI model

class Agent:
    def __init__(self):
        self.n_games = 0  # Count of games played
        self.epsilon = 0  # Randomness factor for exploration vs exploitation
        self.gamma = 0.9  # Discount factor for future rewards
        self.memory = deque(maxlen=MAX_MEMORY)  # Stores past experiences, removes oldest if full
        self.model = Linear_QNet(11, 256, 3)  
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)  

    def get_state(self, game):
        """ Extracts the current game state and returns it as a NumPy array. """
        
        # Get the position of the snake's head
        head = game.snake[0]

        # Define positions around the head (left, right, up, down)
        point_l = Point(head.x - 20, head.y)  # Left of the head
        point_r = Point(head.x + 20, head.y)  # Right of the head
        point_u = Point(head.x, head.y - 20)  # Above the head
        point_d = Point(head.x, head.y + 20)  # Below the head

        # Get the snake's current direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Define the game state as a list of booleans (1 or 0)
        state = [
            # Danger directly ahead
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger to the right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger to the left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Current movement direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location relative to snake
            game.food.x < game.head.x,  # Food is to the left
            game.food.x > game.head.x,  # Food is to the right
            game.food.y < game.head.y,  # Food is above
            game.food.y > game.head.y   # Food is below
        ]

        return np.array(state, dtype=int)  # Convert to NumPy array for easy processing

    def remember(self, state, action, reward, next_state, game_over):
        """ Stores an experience in memory for later training. """
        self.memory.append((state, action, reward, next_state, game_over))  # Removes old experiences if memory is full

    def train_long_memory(self):
        """ Trains the AI using multiple past experiences (batch learning). """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # Get a random batch of past experiences
        else:
            mini_sample = self.memory  # Use all experiences if less than batch size

        # Unpack batch into separate lists
        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        
        # Train using batch data
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)  

    def train_short_memory(self, state, action, reward, next_state, game_over):
        """ Trains the AI immediately after every move (fast learning). """
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        """ Chooses an action using an exploration vs exploitation strategy. """
        
        # Adjust randomness based on the number of games played
        self.epsilon = 80 - self.n_games  # More randomness at the beginning, less over time
        final_move = [0, 0, 0]  # Default: No movement selected

        if random.randint(0, 200) < self.epsilon:  # Exploration: Pick a random move
            move = random.randint(0, 2)  # Randomly choose left (0), right (1), or straight (2)
            final_move[move] = 1  # Activate the chosen move
        else:  # Exploitation: Use the trained model to choose the best move
            state0 = torch.tensor(state, dtype=torch.float)  # Convert state to tensor
            prediction = self.model(state0)  # Get predictions from the neural network
            move = torch.argmax(prediction).item()  # Choose the move with the highest predicted value
            final_move[move] = 1  # Activate the best move

        return final_move  # Return the chosen move as a one-hot encoded list

def train():
    plot_scores=[]
    plot_mean_scores=[]
    total_score= 0
    """ Main function to train the AI by playing multiple games. """
    record = 0  # Keep track of the highest score
    agent = Agent()  # Create the AI agent
    game = SnakeGameAI()  # Create the Snake game instance

    while True:  # Loop to play multiple games continuously
        # Get current game state
        state_old = agent.get_state(game)

        # AI selects a move
        final_move = agent.get_action(state_old)

        # Perform the chosen move and get new game state
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short-term memory (fast learning after each move)
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

        # Store the experience for later use in batch training
        agent.remember(state_old, final_move, reward, state_new, game_over)

        # If game is over, reset and train with long-term memory
        if game_over:
            game.reset()
            agent.n_games += 1  # Increase the number of games played
            agent.train_long_memory()  # Train using stored experiences

            # If this game's score is a new record, update and save the model
            if score > record:
                record = score
                # Save the trained model
                agent.model.save()  

            # Print AI progress
            print('Game', agent.n_games, 'Score:', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score= total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores , plot_mean_scores)









if __name__ == '__main__':
    train()            
