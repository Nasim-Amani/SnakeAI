import pygame  # 🕹️ This is our game library!
import random  # 🎲 Helps us place food in random locations.
from enum import Enum  # 🏷️ This lets us name directions (UP, DOWN, LEFT, RIGHT).
from collections import namedtuple  # 📦 Helps us store (x, y) positions neatly.
import numpy as np  # 🧠 AI will use this for decision-making later.

# 🎬 Step 1: Initialize Pygame
pygame.init()

# 🖋️ Step 2: Define a font (for showing the score)
font = pygame.font.SysFont('arial', 25)  # 🏆 This will help us display the score later!

# 🏹 Step 3: Create an Enum for Directions (for easy movement control)
class Direction(Enum):
    RIGHT = 1  # ➡️ Move Right
    LEFT = 2   # ⬅️ Move Left
    UP = 3     # ⬆️ Move Up
    DOWN = 4   # ⬇️ Move Down

# 📌 Step 4: Define a Point (Stores (x, y) positions)
Point = namedtuple('Point', 'x, y')  # 🏠 Like a home address for each object!

# 🎨 Step 5: Define Colors (for UI elements)
WHITE = (255, 255, 255)  # ⚪ Text color
RED = (200, 0, 0)  # 🍎 Food color
BLUE1 = (0, 0, 255)  # 🟦 Outer Snake Body
BLUE2 = (0, 100, 255)  # 🟦 Inner Snake Body
BLACK = (0, 0, 0)  # 🖤 Background color

# 🎮 Step 6: Game Constants
BLOCK_SIZE = 20  # 🟩 Size of each grid cell
SPEED = 40  # ⚡ Game speed (higher = faster snake)

# 🏗 Step 7: Creating the Game Class
class SnakeGameAI:
    def __init__(self, w=640, h=480):  
        self.w = w
        self.h = h
        # Initialize game window
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI')
        self.clock = pygame.time.Clock()
        self.reset()  # 🔄 Reset game state

    def reset(self):  
        """Resets the game state"""
        self.direction = Direction.RIGHT  # What should the initial direction be?
        self.head = Point(self.w / 2, self.h / 2)  # 🏠 Snake starts in the middle
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        '''
        Start Position (snake facing right) → 🐍
        +----+----+----+
        |  S |  S |  H |  -> (RIGHT)
        +----+----+----+
        '''
        self.score = 0  #What should the initial direction be?
        self.food = None  
        self._place_food()  # 🍏 Place the first food on the screen!
        self.frame_iteration = 0  # 📊 Track how long the snake survives

    # 🍏 Step 8: Adding Food to the Game
    def _place_food(self):
        """Places food at a random position"""
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE # 🎯 Generate a random X position
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE  # 🎯 Generate a random Y position
        self.food = Point(x, y)

        # 🛑 Make sure food doesn't appear inside the snake's body!
        if self.food in self.snake:
            self._place_food  # 🎯 What should we do if food is inside the snake?

    # 🎮 Step 9: Playing the Game (Main Loop)
    def play_step(self, action):
        """Executes one step in the game"""
        self.frame_iteration += 1  # ⏳ Count frames (for AI training)

        # 1️⃣ Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2️⃣ Move the snake
        self._move(action)  # 📝 What function moves the snake?
        self.snake.insert(0, self.head)  # 📝 What should be inserted at the front of the snake’s body?

        # 3️⃣ Check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True  
            reward = -10  # 🎯 What is the penalty for losing?
            return reward, game_over, self.score

        # 4️⃣ Check if food is eaten
        if self.head == self.food:
            self.score += 1 # 🎯 What should we add to increase the score?
            reward = +10  # 🎯 What is the reward for eating food?
            self._place_food()  # 🎯  What function places new food on the screen?
        else:
            self.snake.pop()  # 🐍 Remove tail if food wasn’t eaten
        
        # 5️⃣ Update the UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6️⃣ Return game over status and score
        return reward, game_over, self.score

    # 🚧 Step 10: Collision Detection
    def is_collision(self, pt=None):
        """Checks if the snake collides with the wall or itself"""
        if pt is None:
            pt = self.head  # 🎯 Default to the snake’s head

        # 🛑 Hits boundary?
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True  # 🎯 What happens if it hits the wall?

        # 🛑 Hits itself?
        if pt in self.snake[1:]:
            return True  # 🎯 What happens if the snake bites itself?

        return False
        
    # 🎨 Step 11: Updating the UI - Making Everything Visible!
    def _update_ui(self):
        """Draws the game elements on the screen (Snake, Food, and Score)"""

        # 🖤 1️⃣ Clear the screen by filling it with black
        # This removes the previous frame to prevent a "trailing" effect
        self.display.fill(BLACK)  

        # 🐍 2️⃣ Draw the Snake on the screen
        for pt in self.snake:
            # 🔹 Outer snake body (A blue rectangle)
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            
            # 🔹 Inner shading for a cool 3D effect (A smaller blue rectangle inside)
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # 🍏 3️⃣ Draw the Food
        # This creates a red square at the food's location
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # 🏆 4️⃣ Display the Score
        # 🎯 Render the text "Score: X" in white color
        text = font.render("Score: " + str(self.score), True, WHITE)
        
        # 🎯 Place the text in the top-left corner of the screen
        self.display.blit(text, [0, 0])

        # 🔄 5️⃣ Refresh the screen so all the new elements are visible
        pygame.display.flip()

    # 🐍 Step 12: Moving the Snake
    def _move(self, action):
        """Moves the snake based on AI action.
        
        🕹️ AI can choose one of these 3 actions:
        [1, 0, 0]  → Go straight 🚀 
        [0, 1, 0]  → Turn right 🔄  
        [0, 0, 1]  → Turn left 🔄  
        """
        
        # 🔄 Define movement directions in a clockwise order
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        
        ''' 🧭 Visual representation:
                [RIGHT, DOWN, LEFT, UP]
                    0      1     2    3

                        UP (↑)          
                LEFT (←)   (→) RIGHT 
                    DOWN (↓)
        '''

        # 🎯 Find the current direction of the snake
        idx = clock_wise.index(self.direction)  

        # 🎮 AI's Decision: Does the snake turn or go straight?
        if np.array_equal(action, [1, 0, 0]):  
            # 🚀 AI says "Go straight!" (No change)
            new_dir = clock_wise[idx]  

        elif np.array_equal(action, [0, 1, 0]):  
            # 🔄 AI says "Turn Right!"
            next_idx = (idx + 1) % 4  # Move one step forward in clock_wise list
            new_dir = clock_wise[next_idx]  

            '''
            🔀 Right Turn Mapping:
            If facing RIGHT (0) → Turns to DOWN (1) ⬇️
            If facing DOWN (1) → Turns to LEFT (2) ⬅️
            If facing LEFT (2) → Turns to UP (3) ⬆️
            If facing UP (3) → Turns to RIGHT (0) ➡️
            '''

        else:  # [0, 0, 1]
            # 🔄 AI says "Turn Left!"
            next_idx = (idx - 1) % 4  # Move one step backward in clock_wise list
            new_dir = clock_wise[next_idx]  

            '''
            🔀 Left Turn Mapping:
            If facing RIGHT (0) → Turns to UP (3) ⬆️
            If facing UP (3) → Turns to LEFT (2) ⬅️
            If facing LEFT (2) → Turns to DOWN (1) ⬇️
            If facing DOWN (1) → Turns to RIGHT (0) ➡️
            '''

        # ✅ Update the new direction of the snake
        self.direction = new_dir  

        # 🏁 Move the snake’s head in the new direction
        x = self.head.x  
        y = self.head.y  

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE  # ➡️ Move right
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE  # ⬅️ Move left
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE  # ⬇️ Move down
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE  # ⬆️ Move up

        # 📍 Update the snake’s new head position
        self.head = Point(x, y)
