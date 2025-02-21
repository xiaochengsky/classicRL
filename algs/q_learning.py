# -*- coding: utf-8 -*-
# @Time: 2025/2/21 下午2:53
# @Author: YANG.C
# @File: q_learning.py

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


class DQNetwork(nn.Module):
    def __init__(self):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 200)
        self.fc2 = nn.Linear(200, 300)
        self.fc3 = nn.Linear(300, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def q_learning_on_policy(env, start_state=(0, 0), alpha=0.1, gamma=0.9, epsilon=0.1, iterations=100):
    q = np.zeros((env.num_states, len(env.action_space)))
    v = np.zeros(env.num_states)
    policy = np.random.rand(env.num_states, len(env.action_space))
    policy = policy / policy.sum(axis=1)[:, np.newaxis]

    for k in tqdm(range(iterations)):
        state = start_state
        s = state[1] * env.env_size[0] + state[0]
        a = np.random.choice(np.arange(len(env.action_space)), p=policy[s])
        action = env.action_space[a]
        while state != (4, 4):
            next_state, reward = env.get_next_state_reward(state, action)
            q[s, a] = q[s, a] - alpha * (
                    q[s, a] - (reward + gamma * np.max(q[next_state])))  # solving the Bellman optimality equation

            # update the policy
            idx = np.argmax(q[s])
            policy[s, idx] = 1 - epsilon * (len(env.action_space) - 1) / len(env.action_space)
            policy[s, np.arange(len(env.action_space)) != idx] = epsilon / len(env.action_space)
            v[s] = np.sum(policy[s] * q[s])

            s = next_state
            state = (s % env.env_size[0], s // env.env_size[0])
            a = np.random.choice(np.arange(len(env.action_space)), p=policy[s])
            action = env.action_space[a]

    return v, policy


def dqn(env, gt_state_values, c_steps=5, num_experience=1000, alpha=0.01, gamma=0.9, batch_size=100, epochs=1000):
    # generate experience by the behavior policy and store them in the reply buffer
    policy = np.ones((env.num_states, len(env.action_space)))
    policy = policy / len(env.action_space)
    reply_buffer = []
    s = np.random.choice(env.num_states)
    visit_history = []
    losses = []
    deltas = []

    # generate data for replay buffer
    for e in range(num_experience):
        a = np.random.choice(np.arange(len(env.action_space)), p=policy[s])
        action = env.action_space[a]
        next_state, reward = env.get_next_state_reward((s % env.env_size[0], s // env.env_size[0]), action)
        reply_buffer.append(((s % env.env_size[0], s // env.env_size[0]), a, reward,
                             (next_state % env.env_size[0], next_state // env.env_size[0])))
        visit_history.append((s, a))
        s = next_state

    main_net, target_net = DQNetwork(), DQNetwork()
    # synchronize
    target_net.load_state_dict(main_net.state_dict())
    optimizer = optim.Adam(main_net.parameters(), lr=alpha)
    criterion = nn.MSELoss()
    for k in range(epochs):
        # mini-batch
        batch = np.random.choice(len(reply_buffer), batch_size)  # uniform distribution
        states = torch.tensor([reply_buffer[i][0] for i in batch], dtype=torch.float).view(-1, 2)
        actions = torch.tensor([reply_buffer[i][1] for i in batch], dtype=torch.long).view(-1, 1)
        rewards = torch.tensor([reply_buffer[i][2] for i in batch], dtype=torch.float).view(-1, 1)
        next_states = torch.tensor([reply_buffer[i][3] for i in batch], dtype=torch.float).view(-1, 2)

        main_net_input = states / env.env_size[0]
        q_values = main_net(main_net_input).gather(1, actions.long())

        target_net_input = next_states / env.env_size[0]
        with torch.no_grad():
            next_q_values = target_net(target_net_input).max(dim=1)[0].view(-1, 1)

        target = rewards + gamma * next_q_values

        # main_result vs. gamma * target_result + reward
        loss = criterion(q_values, target)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # synchronize
        if (k + 1) % c_steps == 0:
            target_net.load_state_dict(main_net.state_dict())

        delta = 0
        for s in range(env.num_states):
            with torch.no_grad():
                q_values = main_net(
                    torch.tensor(np.array([s % env.env_size[0], s // env.env_size[0]]) / env.env_size[0],
                                 dtype=torch.float).view(-1, 2))
            delta = max(delta, torch.abs(gt_state_values[s] - torch.max(q_values)).item())
        deltas.append(delta)

    # generate the target policy and target state values
    policy_matrix = np.zeros((env.num_states, len(env.action_space)))
    value_matrix = np.zeros(env.num_states)
    for s in range(env.num_states):
        with torch.no_grad():
            q_values = main_net(torch.tensor(np.array([s % env.env_size[0], s // env.env_size[0]]) / env.env_size[0],
                                             dtype=torch.float).view(-1, 2))
        value_matrix[s] = torch.max(q_values).item()
        idx = torch.argmax(q_values).item()
        policy_matrix[s, idx] = 1

    plt.subplots(3, 1, figsize=(10, 10))
    plt.subplot(3, 1, 1)
    plt.hist([i[0] for i in visit_history])
    plt.subplot(3, 1, 2)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.subplot(3, 1, 3)
    plt.plot(deltas)
    plt.title('Value Error')
    plt.show()

    return value_matrix, policy_matrix
