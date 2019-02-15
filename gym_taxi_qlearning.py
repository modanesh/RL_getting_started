import gym
import numpy as np

env = gym.make("Taxi-v2")
init_state = env.reset()
env.render()

num_states = env.observation_space.n
num_actions = env.action_space.n

print("Taxi-v2")
print(init_state)
print(num_states)
print(num_actions)

env.env.s = 114
env.render()
resutls = env.step(1)
print(resutls)
env.render()


q_table = np.zeros([num_states, num_actions])
G = 0
learning_rate = 0.618

counter = 0
reward = 0
for episode in range(11001):
    done = False
    G, reward = 0, 0
    state = env.reset()
    while done != True:
        action = np.argmax(q_table[state])
        state2, reward, done, info = env.step(action)
        q_table[state, action] += learning_rate * (reward + np.max(q_table[state2]) - q_table[state, action])
        G += reward
        state = state2
    if episode % 50 == 0:
        print('Episode {} Total Reward: {}'.format(episode,G))