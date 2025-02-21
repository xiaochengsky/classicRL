# -*- coding: utf-8 -*-
# @Time: 2025/2/20 下午6:52
# @Author: YANG.C
# @File: policy_iteration.py


import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from example.arguments import args


def policy_iter(env, gamma=0.9, theta=1e-5):
    """
    :param env:
    :param gamma:
    :param theta:
    :return:
    """
    state_values = np.random.uniform(args.reward_forbidden, args.reward_target, env.num_states)
    policy = np.zeros((env.num_states, len(env.action_space)))
    iter_cnts = 0
    while True:
        iter_cnts += 1
        copy_v = state_values.copy()
        for s in range(env.num_states):
            cur_state = (s % env.env_size[0], s // env.env_size[0])
            q_value = []
            for a, action in enumerate(env.action_space):
                next_state, reward = env.get_next_state_reward(cur_state, action)
                # q = immediate reward + v[s']
                q_value.append(reward + gamma * state_values[next_state])
            state_values[s] = np.dot(policy[s], np.array(q_value))

        for s in range(env.num_states):
            cur_state = (s % env.env_size[0], s // env.env_size[0])
            q_value = []
            for a, action in enumerate(env.action_space):
                next_state, reward = env.get_next_state_reward(cur_state, action)
                q_value.append(reward + gamma * state_values[next_state])

            max_action_idx = np.argmax(q_value)
            policy[s, max_action_idx] = 1
            policy[s, np.arange(len(env.action_space)) != max_action_idx] = 0

            if cur_state == env.target_state:
                policy[s, -1] = 1
                policy[s, :-1] = 0

        diff = max(np.abs(copy_v - state_values))
        print(f'Iteration [{iter_cnts}], diff: {diff}')
        if diff < theta:
            break
    return state_values, policy
