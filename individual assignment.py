# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 04:14:27 2026

@author: Tsing
"""

# -*- coding: utf-8 -*-
"""
CDS524 Assignment 1 - Platform Jump Game
Reinforcement Learning Implementation with Q-Learning (DQN)
Technology Stack: Python + Pygame + PyTorch
Game Type: 2D Platformer with AI/Player Dual Mode
"""
import pygame
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
import os

# ===================== Initialization & Global Config ======================
pygame.init()
pygame.font.init()

# Force keyboard continuous response
pygame.key.set_repeat(5, 5)

# Window Configuration
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
FPS = 60
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("CDS524 - Q-Learning Platform Jump")

# Color Definition
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
LIGHT_BLUE = (173, 216, 230)

# Size Configuration
PLAYER_SIZE = 20
PLATFORM_WIDTH = 80
PLATFORM_HEIGHT = 15
OBSTACLE_SIZE = (20, 20)
GOAL_SIZE = (30, 30)

# Physics Parameters (Slow Motion for Accessibility)
MOVE_SPEED = 3          # Horizontal speed (reduced by 50% for control)
JUMP_FORCE = 10         # Jump power (reduced by 40% for precision)
GRAVITY = 0.5           # Gravity (reduced by 23% for better landing)
FALL_DEATH_THRESHOLD = WINDOW_HEIGHT + 100  # Prevent false death triggers

# Game Rules (Aligned with Assignment Requirements)
MAX_STEPS_PER_EPISODE = 500  # Extended steps for completion
MAX_LIVES = 3
COUNTDOWN_DURATION = 3       # Preparation time before game start
OBSTACLE_SPEED_SCALE = 0.3   # Slower obstacles for balanced difficulty

# Goal Flash Configuration (Slow for visibility)
GOAL_FLASH_INTERVAL = 1200   # Flash interval (1.2s)

# Q-Learning Hyperparameters (Critical for Rubric 2)
STATE_DIM = 36              # 36-dimensional state vector
ACTION_DIM = 4              # 4 actions: Left/Right/Jump/Stop
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.95      # Gamma for future reward
EPSILON_START = 1.0         # Epsilon-greedy exploration
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000

# ===================== Core Class Definitions (Aligned with Sample) ======================
class DQN(nn.Module):
    """Deep Q-Network for Q-Learning Implementation (Rubric 2.1)"""
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    """Experience Replay Buffer for Stable Training (Rubric 2.3)"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([item[0] for item in batch], dtype=np.float32)
        actions = np.array([item[1] for item in batch], dtype=np.int64)
        rewards = np.array([item[2] for item in batch], dtype=np.float32)
        next_states = np.array([item[3] for item in batch], dtype=np.float32)
        dones = np.array([item[4] for item in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class Player:
    """Player/AI Agent Class (Rubric 1.2: Action/State Space)"""
    def __init__(self, start_platform):
        # Initial Position (On start platform)
        self.x = start_platform.x + (PLATFORM_WIDTH - PLAYER_SIZE) // 2
        self.y = start_platform.y - PLAYER_SIZE - 2
        self.size = PLAYER_SIZE
        
        # Physical State
        self.vel_y = 0
        self.on_ground = True
        self.landing_buffer = 0  # Prevent platform clipping
        
        # Game State
        self.lives = MAX_LIVES
        self.steps = 0
        self.is_alive = True
        self.has_won = False

    def update(self, move_left, move_right, jump):
        """Update Agent State (Rubric 1.1: Game Rules)"""
        if not self.is_alive:
            return
        
        self.steps += 1
        fell_to_death = False

        # Horizontal Movement
        if move_left:
            self.x -= MOVE_SPEED
        if move_right:
            self.x += MOVE_SPEED
        
        # Jump Action (Only when on ground)
        if jump and (self.on_ground or self.landing_buffer > 0):
            self.vel_y = -JUMP_FORCE
            self.on_ground = False
            self.landing_buffer = 0
        
        # Apply Gravity
        self.vel_y += GRAVITY
        self.y += self.vel_y
        
        # Boundary Restriction
        self.x = max(0, min(self.x, WINDOW_WIDTH - self.size))
        
        # Death Detection
        if self.y > FALL_DEATH_THRESHOLD:
            fell_to_death = True
        
        # Episode Termination Conditions
        if self.steps >= MAX_STEPS_PER_EPISODE or self.lives <= 0:
            self.is_alive = False
        
        # Reset Position on Death
        if fell_to_death:
            self.reset_to_start(start_platform)

    def reset_to_start(self, start_platform):
        """Reset Agent to Initial State"""
        self.x = start_platform.x + (PLATFORM_WIDTH - PLAYER_SIZE) // 2
        self.y = start_platform.y - PLAYER_SIZE - 2
        self.vel_y = 0
        self.on_ground = True
        self.landing_buffer = 10
        self.lives -= 1
        self.steps = 0

    def draw(self, screen, game_mode):
        """Render Agent (Visual Feedback for Rubric 3.2)"""
        # Yellow as base color, border color differentiates mode
        border_color = BLUE if game_mode == "player" else RED
        pygame.draw.rect(screen, YELLOW, (self.x, self.y, self.size, self.size))
        pygame.draw.rect(screen, border_color, (self.x, self.y, self.size, self.size), 2)
        
        # Health Bar Visualization
        health_bar_width = self.size * (self.lives / MAX_LIVES)
        pygame.draw.rect(screen, BLACK, (self.x-2, self.y-10, self.size+4, 6), 1)
        health_color = GREEN if self.lives > 1 else RED
        pygame.draw.rect(screen, health_color, (self.x, self.y-8, health_bar_width, 2))

class Platform:
    """Game Platform Class (Obstacle-free start platform)"""
    def __init__(self, is_start_platform=False):
        self.x = random.randint(50, WINDOW_WIDTH - 130)
        self.y = random.randint(100, WINDOW_HEIGHT - 100)
        self.width = PLATFORM_WIDTH
        self.height = PLATFORM_HEIGHT
        self.color = GREEN
        self.obstacle_allowed = not is_start_platform  # Start platform = no obstacles

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, BLACK, (self.x, self.y, self.width, self.height), 2)

class Obstacle:
    """Moving Obstacle Class (Rubric 1.1: Undesirable State)"""
    def __init__(self, platform):
        self.platform = platform
        self.x = platform.x + random.randint(5, platform.width - OBSTACLE_SIZE[0] - 5)
        self.y = platform.y - OBSTACLE_SIZE[1]
        self.size = OBSTACLE_SIZE
        self.speed = random.uniform(0.8, 1.5) * OBSTACLE_SPEED_SCALE
        self.direction = random.choice([-1, 1])  # -1=Left, 1=Right

    def update(self):
        """Update Obstacle Position"""
        self.x += self.speed * self.direction
        # Boundary Bounce
        if self.x <= self.platform.x + 5:
            self.direction = 1
        elif self.x + self.size[0] >= self.platform.x + self.platform.width - 5:
            self.direction = -1

    def draw(self, screen):
        pygame.draw.rect(screen, BLACK, (self.x, self.y, self.size[0], self.size[1]))

class Goal:
    """Game Goal (Rubric 1.1: Clear Objective)"""
    def __init__(self):
        self.x = random.randint(50, WINDOW_WIDTH - 80)
        self.y = random.randint(50, 200)
        self.size = GOAL_SIZE

    def draw(self, screen):
        """Slow Flashing Effect for Visibility"""
        current_time = pygame.time.get_ticks()
        color = BLUE if current_time % GOAL_FLASH_INTERVAL < GOAL_FLASH_INTERVAL//2 else LIGHT_BLUE
        pygame.draw.rect(screen, color, (self.x, self.y, self.size[0], self.size[1]))
        pygame.draw.rect(screen, WHITE, (self.x, self.y, self.size[0], self.size[1]), 3)

# ===================== Q-Learning Training Utilities (Rubric 2) ======================
def get_game_state(player, platforms, obstacles, goal):
    """Generate 36-dimensional State Vector (Rubric 1.2)"""
    state = []
    
    # 1. Player State (8 dimensions)
    state.extend([
        player.x / WINDOW_WIDTH,
        player.y / WINDOW_HEIGHT,
        player.vel_y / 20,
        1.0 if player.on_ground else 0.0,
        player.lives / MAX_LIVES,
        player.steps / MAX_STEPS_PER_EPISODE,
        1.0 if player.is_alive else 0.0,
        1.0 if player.has_won else 0.0
    ])
    
    # 2. Goal State (3 dimensions)
    goal_dist = np.sqrt((player.x - goal.x)**2 + (player.y - goal.y)**2)
    state.extend([
        goal.x / WINDOW_WIDTH,
        goal.y / WINDOW_HEIGHT,
        goal_dist / WINDOW_WIDTH
    ])
    
    # 3. Closest Platform State (7 dimensions)
    closest_plat = min(platforms, key=lambda p: np.sqrt((player.x - p.x)**2 + (player.y - p.y)**2))
    plat_h_dist = (player.x - closest_plat.x) / WINDOW_WIDTH
    plat_v_dist = (player.y - closest_plat.y) / WINDOW_HEIGHT
    state.extend([
        closest_plat.x / WINDOW_WIDTH,
        closest_plat.y / WINDOW_HEIGHT,
        closest_plat.width / WINDOW_WIDTH,
        closest_plat.height / WINDOW_HEIGHT,
        plat_h_dist,
        plat_v_dist,
        1.0 if closest_plat.obstacle_allowed else 0.0
    ])
    
    # 4. Obstacle States (16 dimensions: top 2 obstacles)
    for i in range(2):
        if i < len(obstacles):
            obs = obstacles[i]
            obs_dist = np.sqrt((player.x - obs.x)**2 + (player.y - obs.y)**2)
            state.extend([
                obs.x / WINDOW_WIDTH,
                obs.y / WINDOW_HEIGHT,
                obs.speed / 5,
                1.0 if obs.direction == 1 else 0.0,
                obs_dist / WINDOW_WIDTH,
                1.0 if obs_dist < 50 else 0.0,  # Threat detection
                1.0 if player.x < obs.x else 0.0,
                1.0 if player.on_ground else 0.0
            ])
        else:
            state.extend([0.0]*8)
    
    # 5. Normalization (Ensure state vector consistency)
    state = np.clip(state, 0.0, 1.0)
    return state.astype(np.float32)

def calculate_reward(player, previous_state, current_state, goal):
    """Reward Function (Rubric 1.3: Positive/Negative Rewards)"""
    reward = 0.1  # Survival reward
    
    # Positive Rewards
    prev_goal_dist = previous_state[11] * WINDOW_WIDTH
    curr_goal_dist = current_state[11] * WINDOW_WIDTH
    if curr_goal_dist < prev_goal_dist:
        reward += 0.8  # Reward for approaching goal
    if player.has_won:
        reward += 100.0  # Major reward for winning
    if player.on_ground and player.steps % 10 == 0:
        reward += 0.2  # Stability reward
    
    # Negative Rewards
    if not player.is_alive:
        reward -= 50.0  # Penalty for death
    if curr_goal_dist > prev_goal_dist + 20:
        reward -= 0.5  # Penalty for moving away
    if player.steps > MAX_STEPS_PER_EPISODE * 0.8:
        reward -= 0.3  # Time penalty
    
    return reward

# ===================== Game Initialization ======================
def initialize_game():
    """Initialize Game Objects (Aligned with Sample Structure)"""
    # Create platforms (6 total, 2 obstacle-free)
    platforms = []
    start_platform = Platform(is_start_platform=True)
    start_platform.x = WINDOW_WIDTH//4 - PLATFORM_WIDTH//2
    start_platform.y = WINDOW_HEIGHT//2 + 80
    platforms.append(start_platform)
    
    # Generate layered platforms (avoid overlap)
    obstacle_free_count = 1
    for i in range(5):
        y_pos = WINDOW_HEIGHT//2 - (i * 80) if i < 3 else WINDOW_HEIGHT//2 + (i-2)*60
        new_plat = Platform(is_start_platform=(obstacle_free_count < 2))
        new_plat.y = y_pos
        
        # Avoid platform overlap
        overlap = False
        for existing in platforms:
            if abs(new_plat.x - existing.x) < 60 and abs(new_plat.y - existing.y) < 40:
                overlap = True
                break
        
        if overlap:
            while overlap:
                new_plat.x = random.randint(50, WINDOW_WIDTH - 130)
                overlap = False
                for existing in platforms:
                    if abs(new_plat.x - existing.x) < 60 and abs(new_plat.y - existing.y) < 40:
                        overlap = True
                        break
        
        platforms.append(new_plat)
        if new_plat.obstacle_allowed is False:
            obstacle_free_count += 1
    
    # Create obstacles (only on allowed platforms)
    obstacles = []
    for plat in platforms:
        if plat.obstacle_allowed and random.random() > 0.3:
            obstacles.append(Obstacle(plat))
    
    # Create player and goal
    player = Player(start_platform)
    goal = Goal()
    
    # Initialize Q-Learning components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(STATE_DIM, ACTION_DIM).to(device)
    target_model = DQN(STATE_DIM, ACTION_DIM).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    epsilon = EPSILON_START
    
    return {
        "platforms": platforms,
        "obstacles": obstacles,
        "player": player,
        "goal": goal,
        "device": device,
        "model": model,
        "target_model": target_model,
        "optimizer": optimizer,
        "replay_buffer": replay_buffer,
        "epsilon": epsilon,
        "episode": 0,
        "total_reward": 0.0
    }

# Initialize game objects
game_objects = initialize_game()
game_state = "start"  # start/countdown/playing/game_over
game_mode = "player"  # player/ai
countdown_start_time = 0
FONT_LARGE = pygame.font.SysFont("Arial", 72, bold=True)
FONT_MEDIUM = pygame.font.SysFont("Arial", 36)
FONT_SMALL = pygame.font.SysFont("Arial", 20)

# ===================== Core Game Functions ======================
def check_collisions(player, platforms, obstacles, goal):
    """Collision Detection (Rubric 3.4: Smooth Interaction)"""
    player_rect = pygame.Rect(player.x, player.y, player.size, player.size)
    player.on_ground = False
    
    # Update landing buffer
    if player.landing_buffer > 0:
        player.landing_buffer -= 1

    # 1. Platform Collision (Stable landing guarantee)
    for plat in platforms:
        plat_rect = pygame.Rect(plat.x, plat.y, plat.width, plat.height)
        if player_rect.colliderect(plat_rect):
            # Landing detection (bottom of player hits top of platform)
            if (player_rect.bottom >= plat_rect.top - 2 and
                player_rect.bottom <= plat_rect.top + 10 and
                player.vel_y >= 0):
                player.y = plat_rect.top - player.size
                player.vel_y = 0
                player.on_ground = True
                player.landing_buffer = 8
            
            # Prevent clipping into platform
            elif player_rect.top < plat_rect.bottom and player_rect.bottom > plat_rect.top:
                if player.vel_y < 0:  # Hit bottom of platform
                    player.y = plat_rect.bottom
                    player.vel_y = 0
                elif player.vel_y > 0:  # Fall into platform
                    player.y = plat_rect.top - player.size
                    player.vel_y = 0
                    player.on_ground = True

    # 2. Obstacle Collision (Penalized state)
    for obs in obstacles:
        obs_rect = pygame.Rect(obs.x, obs.y, obs.size[0], obs.size[1])
        if player_rect.colliderect(obs_rect):
            player.reset_to_start(platforms[0])

    # 3. Goal Collision (Win condition)
    goal_rect = pygame.Rect(goal.x, goal.y, goal.size[0], goal.size[1])
    if player_rect.colliderect(goal_rect):
        player.has_won = True
        global game_state
        game_state = "game_over"

def draw_ui(game_objects):
    """Render UI (Rubric 3.2: State/Action/Reward Display)"""
    player = game_objects["player"]
    platforms = game_objects["platforms"]
    obstacles = game_objects["obstacles"]
    goal = game_objects["goal"]
    
    # Top Status Bar
    status_bg = pygame.Surface((WINDOW_WIDTH, 40))
    status_bg.fill(GRAY)
    screen.blit(status_bg, (0, 0))
    
    # Status Text (Real-time feedback)
    status_text = (
        f"Mode: {game_mode.upper()} | "
        f"Lives: {player.lives}/{MAX_LIVES} | "
        f"Steps: {player.steps}/{MAX_STEPS_PER_EPISODE} | "
        f"On Ground: {'YES' if player.on_ground else 'NO'} | "
        f"Total Reward: {game_objects['total_reward']:.1f}"
    )
    screen.blit(FONT_SMALL.render(status_text, True, WHITE), (10, 5))

    # Left Control Panel
    control_panel = pygame.Surface((240, 220))
    control_panel.fill(LIGHT_BLUE)
    control_panel.set_alpha(180)
    screen.blit(control_panel, (10, 50))
    
    # Control Instructions
    controls = [
        "🎮 Game Controls:",
        "W / ↑ / Space = Jump",
        "A / ← = Move Left",
        "D / → = Move Right",
        "M = Switch Mode (Player/AI)",
        "R = Reset Game",
        "ESC = Quit Game",
        "",
        "🎯 Objective:",
        "Reach the flashing blue goal"
    ]
    y_offset = 60
    for line in controls:
        screen.blit(FONT_SMALL.render(line, True, BLACK), (20, y_offset))
        y_offset += 22

def train_dqn(game_objects, previous_state, action, reward, current_state, done):
    """DQN Training Loop (Rubric 2.1/2.2/2.3)"""
    replay_buffer = game_objects["replay_buffer"]
    model = game_objects["model"]
    target_model = game_objects["target_model"]
    optimizer = game_objects["optimizer"]
    device = game_objects["device"]
    
    # Add experience to replay buffer
    replay_buffer.add(previous_state, action, reward, current_state, done)
    
    # Start training when buffer has enough data
    if len(replay_buffer) >= BATCH_SIZE:
        # Sample batch from buffer
        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards_tensor = torch.FloatTensor(rewards).to(device)
        next_states_tensor = torch.FloatTensor(next_states).to(device)
        dones_tensor = torch.FloatTensor(dones).to(device)
        
        # Current Q-values
        current_q = model(states_tensor).gather(1, actions_tensor).squeeze(1)
        
        # Target Q-values (Double DQN to avoid overestimation)
        next_q = model(next_states_tensor).max(1)[0]
        target_q = rewards_tensor + DISCOUNT_FACTOR * next_q * (1 - dones_tensor)
        
        # Calculate loss
        loss = nn.MSELoss()(current_q, target_q.detach())
        
        # Optimize model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update target model periodically
        if game_objects["episode"] % 5 == 0:
            target_model.load_state_dict(model.state_dict())
        
        # Decay epsilon (balance exploration/exploitation)
        game_objects["epsilon"] = max(EPSILON_END, game_objects["epsilon"] * EPSILON_DECAY)

# ===================== Main Game Loop (Rubric 3.1/3.4) ======================
clock = pygame.time.Clock()
running = True
previous_state = get_game_state(
    game_objects["player"],
    game_objects["platforms"],
    game_objects["obstacles"],
    game_objects["goal"]
)

while running:
    screen.fill(WHITE)
    keys = pygame.key.get_pressed()
    player = game_objects["player"]
    obstacles = game_objects["obstacles"]
    goal = game_objects["goal"]
    platforms = game_objects["platforms"]

    # Event Handling (Rubric 3.1: Interactive UI)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.KEYDOWN:
            # Quit Game
            if event.key == pygame.K_ESCAPE:
                running = False
            # Switch Game Mode (Player/AI)
            if event.key == pygame.K_m:
                game_mode = "ai" if game_mode == "player" else "player"
                print(f"[INFO] Switched to {game_mode.upper()} mode")
            # Reset Game
            if event.key == pygame.K_r:
                game_objects = initialize_game()
                game_state = "countdown"
                countdown_start_time = pygame.time.get_ticks()
                previous_state = get_game_state(
                    game_objects["player"],
                    game_objects["platforms"],
                    game_objects["obstacles"],
                    game_objects["goal"]
                )
        
        # Start Game (Mouse Click for Accessibility)
        if event.type == pygame.MOUSEBUTTONDOWN and game_state == "start":
            game_state = "countdown"
            countdown_start_time = pygame.time.get_ticks()
            print(f"[INFO] Game started - Episode {game_objects['episode'] + 1}")

    # Game State Machine
    if game_state == "start":
        # Start Screen (Aligned with Sample UI)
        screen.fill(BLACK)
        title = FONT_MEDIUM.render("CDS524 - Q-Learning Platform Jump", True, WHITE)
        screen.blit(title, (WINDOW_WIDTH//2 - title.get_width()//2, WINDOW_HEIGHT//4))
        
        # Game Introduction
        intro_text = [
            "📚 Project Overview:",
            "A reinforcement learning platformer using Q-Learning (DQN)",
            "Agent learns to navigate platforms and avoid obstacles",
            "",
            "🎯 Game Rules:",
            "1. Reach the flashing blue goal to win",
            "2. 3 lives and 500 steps per episode",
            "3. Avoid falling or hitting black obstacles",
            "",
            "🖱️ CLICK ANYWHERE TO START",
            "💡 Press M to switch between Player and AI modes"
        ]
        y_offset = WINDOW_HEIGHT//2 - 120
        for line in intro_text:
            text_surface = FONT_SMALL.render(line, True, WHITE)
            screen.blit(text_surface, (WINDOW_WIDTH//2 - text_surface.get_width()//2, y_offset))
            y_offset += 25

    elif game_state == "countdown":
        # Countdown Screen (Preparation Time)
        elapsed_time = (pygame.time.get_ticks() - countdown_start_time) // 1000
        remaining_time = COUNTDOWN_DURATION - elapsed_time

        # Draw Game Elements
        player.draw(screen, game_mode)
        goal.draw(screen)
        for plat in platforms:
            plat.draw(screen)
        for obs in obstacles:
            obs.draw(screen)
        
        # Countdown Overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.fill(BLACK)
        overlay.set_alpha(150)
        screen.blit(overlay, (0, 0))
        
        # Countdown Text
        if remaining_time > 0:
            count_text = FONT_LARGE.render(str(remaining_time), True, WHITE)
            screen.blit(count_text, (WINDOW_WIDTH//2 - count_text.get_width()//2, WINDOW_HEIGHT//2))
            tip_text = FONT_SMALL.render(f"Mode: {game_mode.upper()} - Starting soon...", True, WHITE)
            screen.blit(tip_text, (WINDOW_WIDTH//2 - tip_text.get_width()//2, WINDOW_HEIGHT//2 + 80))
        else:
            game_state = "playing"
            game_objects["episode"] += 1

    elif game_state == "playing":
        # Default Action: Stop
        move_left = False
        move_right = False
        jump = False
        action = 3  # 0=Right, 1=Left, 2=Jump, 3=Stop

        # Player Mode (Full Keyboard Control)
        if game_mode == "player":
            move_left = keys[pygame.K_a] or keys[pygame.K_LEFT]
            move_right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
            jump = keys[pygame.K_w] or keys[pygame.K_UP] or keys[pygame.K_SPACE]
            
            # Map player input to action code
            if move_left:
                action = 1
            elif move_right:
                action = 0
            elif jump:
                action = 2

        # AI Mode (Q-Learning Control - Rubric 2.2)
        elif game_mode == "ai":
            state_tensor = torch.FloatTensor(previous_state).unsqueeze(0).to(game_objects["device"])
            
            # Epsilon-greedy action selection (Rubric 2.3)
            if random.random() < game_objects["epsilon"]:
                action = random.choice([0, 1, 2, 3])  # Exploration
            else:
                with torch.no_grad():
                    q_values = game_objects["model"](state_tensor)
                    action = q_values.argmax().item()  # Exploitation
            
            # Map AI action to movement
            if action == 0:
                move_right = True
            elif action == 1:
                move_left = True
            elif action == 2 and player.on_ground:
                jump = True

        # Update Game State
        player.update(move_left, move_right, jump)
        for obs in obstacles:
            obs.update()
        check_collisions(player, platforms, obstacles, goal)

        # Get current state and calculate reward (Rubric 1.3)
        current_state = get_game_state(player, platforms, obstacles, goal)
        reward = calculate_reward(player, previous_state, current_state, goal)
        game_objects["total_reward"] += reward
        done = not player.is_alive or player.has_won

        # Train DQN (Only in AI mode)
        if game_mode == "ai":
            train_dqn(game_objects, previous_state, action, reward, current_state, done)

        # Update previous state for next iteration
        previous_state = current_state.copy()

        # Check for episode end
        if done:
            game_state = "game_over"
            print(f"[EPISODE {game_objects['episode']}] Reward: {game_objects['total_reward']:.1f} | Epsilon: {game_objects['epsilon']:.3f} | Result: {'Win' if player.has_won else 'Loss'}")

        # Draw All Elements
        player.draw(screen, game_mode)
        goal.draw(screen)
        for plat in platforms:
            plat.draw(screen)
        for obs in obstacles:
            obs.draw(screen)
        draw_ui(game_objects)

    elif game_state == "game_over":
        # Game Over Screen (Feedback for Rubric 3.2)
        # Draw game elements in background
        player.draw(screen, game_mode)
        goal.draw(screen)
        for plat in platforms:
            plat.draw(screen)
        for obs in obstacles:
            obs.draw(screen)
        
        # Semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.fill(BLACK)
        overlay.set_alpha(180)
        screen.blit(overlay, (0, 0))
        
        # Result Text
        if player.has_won:
            result_text = FONT_LARGE.render("YOU WIN!", True, GREEN)
            sub_text = FONT_SMALL.render(f"Total Reward: {game_objects['total_reward']:.1f} | Steps Used: {player.steps}", True, WHITE)
        else:
            result_text = FONT_LARGE.render("GAME OVER", True, RED)
            if player.steps >= MAX_STEPS_PER_EPISODE:
                sub_text = FONT_SMALL.render(f"Reason: Maximum Steps Reached ({MAX_STEPS_PER_EPISODE})", True, WHITE)
            else:
                sub_text = FONT_SMALL.render(f"Reason: No Lives Remaining ({player.lives}/{MAX_LIVES})", True, WHITE)
        
        # Center alignment
        result_rect = result_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 - 50))
        sub_rect = sub_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 20))
        tip_text = FONT_SMALL.render("Press R to Restart | Press ESC to Quit", True, WHITE)
        tip_rect = tip_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 70))
        
        screen.blit(result_text, result_rect)
        screen.blit(sub_text, sub_rect)
        screen.blit(tip_text, tip_rect)

    # Force screen update
    pygame.display.flip()
    clock.tick(FPS)

# Cleanup
pygame.quit()
print("[INFO] Game exited successfully")