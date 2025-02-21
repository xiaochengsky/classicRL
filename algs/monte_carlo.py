# -*- coding: utf-8 -*-
# @Time: 2025/2/20 下午7:54
# @Author: YANG.C
# @File: monte_carlo_greedy.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from example.arguments import args


def menton_calro_basic(env, gamma=0.9, episodes=100, iteration=3):
    """
    :param episodes:
    :param env:
    :param gamma:
    :param episode:
    :param iteration:
    :return:
    """
    # policy = np.zeros((env.num_states, len(env.action_space)))
    policy = np.random.uniform(0, 1, (env.num_states, len(env.action_space)))
    policy[:, 0] = 1
    v = np.zeros(env.num_states)

    # k-th iteration
    for k in tqdm(range(iteration)):
        # for loop each state
        for s in range(env.num_states):
            state = (s % env.env_size[0], s // env.env_size[0])
            q_values = []
            for a, action in enumerate(env.action_space):
                q_value = 0
                tmp_state = state
                # mc base
                for e in range(episodes):
                    next_state, reward = env.get_next_state_reward(tmp_state, action)
                    q_value += (gamma ** e) * reward
                    tmp_state = (next_state % env.env_size[0], next_state // env.env_size[0])
                    action_idx = np.argmax(policy[next_state])
                    action = env.action_space[action_idx]
                q_values.append(q_value)

            # policy improvement
            max_action_value_idx = np.argmax(q_values)
            policy[s, max_action_value_idx] = 1
            policy[s, np.arange(len(env.action_space)) != max_action_value_idx] = 0
            v[s] = q_values[max_action_value_idx]
    return v, policy


def monte_carlo_exploring(env, gamma=0.9, episodes=1000, iteration=1000):
    # deterministic behavior policy
    policy = np.eye(5)[np.random.randint(0, 5, size=(env.env_size[0] * env.env_size[1]))]
    q = np.zeros((env.num_states, len(env.action_space)))
    v = np.zeros(env.num_states)
    return_temp = np.zeros((env.num_states, len(env.action_space)))
    num_visits = np.zeros((env.num_states, len(env.action_space)))

    for i in tqdm(range(iteration)):
        state_action_pairs = []
        rewards = []
        # generate s-a pairs in every-visit mode
        # pair_idx = i % (env.num_states * len(env.action_space))
        # s, a = pair_idx // env.num_states, pair_idx % len(env.action_space)
        s = np.random.randint(0, env.num_states)
        a = np.random.randint(0, len(env.action_space))

        state_action_pairs.append((s, a))
        next_state, reward = env.get_next_state_reward((s % env.env_size[0], s // env.env_size[0]), env.action_space[a])
        rewards.append(reward)
        for e in range(episodes):
            tmp_state = next_state
            # sample
            action_idx = np.argmax(policy[tmp_state])
            action = env.action_space[action_idx]
            # record
            state_action_pairs.append((tmp_state, action_idx))
            next_state, reward = env.get_next_state_reward((tmp_state % env.env_size[0], tmp_state // env.env_size[0]),
                                                           action)
            rewards.append(reward)

        # policy evaluation
        g = 0
        for w in range(len(state_action_pairs) - 1, -1, -1):
            g = gamma * g + rewards[w]
            # this state-action has appeared in previous state-action chain
            # if state_action_pairs[w] in state_action_pairs[:w]:
            s, a = state_action_pairs[w]
            return_temp[s, a] += g
            num_visits[s, a] += 1
            q[s, a] = return_temp[s, a] / num_visits[s, a]

        # policy improvement
        for s in range(env.num_states):
            for a, _ in enumerate(env.action_space):
                q[s, a] = return_temp[s, a]
            max_action_value_idx = np.argmax(q[s])
            policy[s, max_action_value_idx] = 1
            policy[s, np.arange(len(env.action_space)) != max_action_value_idx] = 0
            v[s] = max(q[s])

    return v, policy


def menton_carlo_e_greedy(env, epsilon=0.2, gamma=0.9, episodes=10000, iteration=100):
    """
    :param env:
    :param gamma:
    :param episodes:
    :param iteration:
    :return:
    """
    # stochastic behavior policy
    policy = np.random.rand(env.num_states, len(env.action_space))
    policy /= policy.sum(axis=1)[:, np.newaxis]
    q = np.zeros((env.num_states, len(env.action_space)))
    v = np.zeros(env.num_states)

    for k in tqdm(range(iteration)):
        tmp_return = np.zeros((env.num_states, len(env.action_space)))
        tmp_num = np.zeros((env.num_states, len(env.action_space)))
        num_visits = np.zeros((env.num_states, len(env.action_space)))

        s = 0
        a = np.random.choice(len(env.action_space), p=policy[s])
        pairs = []
        rewards = []
        tmp_state = s
        for i in range(episodes):
            action = env.action_space[a]
            pairs.append((tmp_state, a))
            next_state, reward = env.get_next_state_reward((tmp_state % env.env_size[0], tmp_state // env.env_size[0]),
                                                           action)
            rewards.append(reward)
            tmp_state = next_state
            a = np.random.choice(len(env.action_space), p=policy[tmp_state])

        g = 0
        for t in range(len(rewards) - 1, -1, -1):
            g = gamma * g + rewards[t]
            s, a = pairs[t]
            tmp_return[s, a] += g
            num_visits[s, a] += 1
            q[s, a] = tmp_return[s, a] / num_visits[s, a] if num_visits[s, a] != 0 else 0
            max_action_value_idx = np.argmax(q[s])
            policy[s, max_action_value_idx] = 1 - epsilon * (len(env.action_space) - 1) / len(env.action_space)
            policy[s, np.arange(len(env.action_space)) != max_action_value_idx] = epsilon / len(env.action_space)
            v[s] = q[s, max_action_value_idx]

    return v, policy
