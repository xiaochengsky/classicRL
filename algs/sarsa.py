# -*- coding: utf-8 -*-
# @Time: 2025/2/21 下午1:35
# @Author: YANG.C
# @File: sarsa.py

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def sarsa(env, start_state=(0, 0), gamma=0.9, alpha=0.1, epsilon=0.1, episodes=1000, iterations=100):
    """
    :param env:
    :param start_state:
    :param gamma:
    :param alpha:
    :param epsilon:
    :param episodes:
    :param iterations:
    :return:
    """
    q = np.zeros((env.num_states, len(env.action_space)))
    v = np.zeros(env.num_states)
    policy = np.random.rand(env.num_states, len(env.action_space))
    policy /= policy.sum(axis=1)[:, np.newaxis]

    lengths = []
    total_rewards = []

    for k in tqdm(range(iterations)):
        # init
        state = start_state
        s = state[1] * env.env_size[0] + state[0]
        a = np.random.choice(np.arange(len(env.action_space)), p=policy[s])
        action = env.action_space[a]

        length = 0
        total_reward = 0

        # target state
        while state != (4, 4):
            next_state, reward = env.get_next_state_reward(state, action)
            next_action = np.random.choice(np.arange(len(env.action_space)), p=policy[next_state])

            # update action value
            # TD optimization equation
            q[s, a] = q[s, a] - alpha * (q[s, a] - (reward + gamma * q[next_state, next_action]))

            max_action_idx = np.argmax(q[s])
            # e-greedy
            policy[s, max_action_idx] = 1 - epsilon * (len(env.action_space) - 1) / len(env.action_space)
            policy[s, np.arange(len(env.action_space)) != max_action_idx] = epsilon / len(env.action_space)

            v[s] = np.sum(policy[s] * q[s])
            s = next_state
            state = (next_state % env.env_size[0], next_state // env.env_size[0])
            action = env.action_space[next_action]
            a = next_action
            length += 1
            total_reward += reward
        lengths.append(length)
        total_rewards.append(total_reward)

    fig = plt.subplots(2, 1)
    plt.subplot(2, 1, 1)
    plt.plot(lengths)
    plt.xlabel('Iterations')
    plt.ylabel('Length of episodes')
    plt.subplot(2, 1, 2)
    plt.plot(total_rewards)
    plt.xlabel('Iterations')
    plt.ylabel('Total rewards of episodes')
    plt.show()
    return v, policy


def sarsa_n_step(env, start_state=(0, 0), steps=3, gamma=0.9, alpha=0.1, epsilon=0.1, episodes=1000, iterations=100):
    q = np.zeros((env.num_states, len(env.action_space)))
    v = np.zeros(env.num_states)
    policy = np.random.rand(env.num_states, len(env.action_space))
    policy /= policy.sum(axis=1)[:, np.newaxis]

    for k in tqdm(range(iterations)):
        # init
        state = start_state
        s = state[1] * env.env_size[0] + state[0]
        a = np.random.choice(np.arange(len(env.action_space)), p=policy[s])
        action = env.action_space[a]

        # target state
        while state != (4, 4):
            rewards = 0
            next_state, reward = env.get_next_state_reward(state, action)
            rewards = rewards * gamma + reward
            next_action = env.action_space[np.random.choice(np.arange(len(env.action_space)), p=policy[next_state])]

            # n-step
            for t in range(steps - 2):
                next_state, reward = env.get_next_state_reward(
                    (next_state % env.env_size[0], next_state // env.env_size[0]), next_action)
                rewards = rewards * gamma + reward
                next_action = env.action_space[np.random.choice(np.arange(len(env.action_space)), p=policy[next_state])]

            tmp_next_state = next_state
            tmp_next_action = np.random.choice(np.arange(len(env.action_space)), p=policy[tmp_next_state])
            q[s, a] = q[s, a] - alpha * (q[s, a] - (rewards + (gamma ** steps) * q[tmp_next_state, tmp_next_action]))
            max_action_idx = np.argmax(q[s])
            # e-greedy
            policy[s, max_action_idx] = 1 - epsilon * (len(env.action_space) - 1) / len(env.action_space)
            policy[s, np.arange(len(env.action_space)) != max_action_idx] = epsilon / len(env.action_space)
            s = tmp_next_state
            a = tmp_next_action
            v[s] = np.sum(policy[s] * q[s])
            state = (tmp_next_state % env.env_size[0], tmp_next_state // env.env_size[0])
            action = env.action_space[tmp_next_action]
    return v, policy
