import torch
import random
import numpy as np 
from snake import SnakeGameAI, Direction, Point
from collections import deque
from model import Linear_QNet, QTrainer
from plotter import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.01

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilom = 0 # Randomness
        self.gamma = 0.9 # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)
    def curr_state(self, snake):
        head = snake.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = snake.direction == Direction.LEFT
        dir_r = snake.direction == Direction.RIGHT
        dir_u = snake.direction == Direction.UP
        dir_d = snake.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and snake.is_collision(point_r)) or 
            (dir_l and snake.is_collision(point_l)) or 
            (dir_u and snake.is_collision(point_u)) or 
            (dir_d and snake.is_collision(point_d)),

            # Danger right
            (dir_u and snake.is_collision(point_r)) or 
            (dir_d and snake.is_collision(point_l)) or 
            (dir_l and snake.is_collision(point_u)) or 
            (dir_r and snake.is_collision(point_d)),

            # Danger left
            (dir_d and snake.is_collision(point_r)) or 
            (dir_u and snake.is_collision(point_l)) or 
            (dir_r and snake.is_collision(point_u)) or 
            (dir_l and snake.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            snake.food.x < snake.head.x,  # food left
            snake.food.x > snake.head.x,  # food right
            snake.food.y < snake.head.y,  # food up
            snake.food.y > snake.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember_func(self, state, action, reward, next_state, done):
         self.memory.append((state, action, reward, next_state, done))

    def training_high_mem(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def training_short_mem(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores=[]
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    snake = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.curr_state(snake)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = snake.play_step(final_move)
        state_new = agent.curr_state(snake)

        # train short memory
        agent.training_short_mem(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember_func(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            snake.reset()
            agent.n_games += 1
            agent.training_high_mem()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()


 