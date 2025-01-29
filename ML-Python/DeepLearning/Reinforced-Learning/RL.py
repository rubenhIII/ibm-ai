import gym
import numpy as np

# Create the environment  
env = gym.make('CartPole-v1')

# Set random seed for reproducibility  
np.random.seed(42)
env.reset(seed=42)

# Suppress warnings for a cleaner notebook or console experience
import warnings
warnings.filterwarnings('ignore')

# Disable warnings for a cleaner notebook or console experience
def warn(*args, **kwargs):
    pass
warnings.warn = warn

# Import necessary libraries
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import Adam

def build_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
model = build_model(state_size, action_size)

from collections import deque
import random

memory = deque(maxlen=2000)
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
 
def act(state):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    q_values = model.predict(state)
    return np.argmax(q_values[0])

def replay(batch_size):
    global epsilon
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = reward + gamma * np.amax(model.predict(next_state)[0])
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay 

# Training loop
episodes = 50  # More episodes to ensure sufficient training
batch_size = 32  # Mini-batch size for replay training
gamma = 0.95  # Discount factor for future rewards
 
for e in range(episodes):
    state = env.reset()
    if isinstance(state, tuple):  # Handle tuple output
        state = state[0]
    state = np.reshape(state, [1, state_size])
 
    for time in range(200):  # Max steps per episode
        # Choose action using epsilon-greedy policy
        action = act(state)
 
        # Perform action in the environment
        result = env.step(action)
        if len(result) == 4:  # Handle 4-value output
            next_state, reward, done, _ = result
        else:  # Handle 5-value output
            next_state, reward, done, _, _ = result
 
        if isinstance(next_state, tuple):  # Handle tuple next_state
            next_state = next_state[0]
        next_state = np.reshape(next_state, [1, state_size])
 
        # Store experience in memory
        remember(state, action, reward, next_state, done)
 
        # Update state
        state = next_state
 
        if done:  # If episode ends
            print(f"Episode: {e+1}/{episodes}, Score: {time}, Epsilon: {epsilon:.2}")
            break
 
    # Train the model using replay memory
    if len(memory) > batch_size:
        replay(batch_size)
 
env.close()