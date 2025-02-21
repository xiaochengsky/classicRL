# -*- coding: utf-8 -*-
# @Time: 2025/2/20 下午4:03
# @Author: YANG.C
# @File: value_iteration.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Tuple

from example.arguments import args


def value_iter(env, gamma=0.9, theta=1e-5) -> Tuple[np.array, np.array]:
    """
    value iteration for solving the Bellman Optimization Equation (BOE)
    :param env: interaction env
    :param gamma: discount rate
    :param theta: threshold
    :return:
    """
    state_values = np.random.uniform(args.reward_forbidden, args.reward_target, env.num_states)
    policy = np.zeros((env.num_states, len(env.action_space)))

    print(state_values.shape)
    print(policy.shape)
    print(args)
    print(env.num_states, env.action_space, env.env_size)
    # exit()

    iter_cnts = 0
    while True:
        iter_cnts += 1
        diff = 0

        for s in range(env.num_states):
            # current state (0, 0)
            cur_state = (s % env.env_size[0], s // env.env_size[0])
            # current value_state
            v = state_values[s]
            q_value = []

            for a, action in enumerate(env.action_space):
                next_state, reward = env.get_next_state_reward(cur_state, action)
                # q = immediate reward + v[s']
                q_value.append(reward + gamma * state_values[next_state])

            # max action value, a*(s) = argmax_a[q(s, a)]
            max_action_idx = np.argmax(q_value)

            # policy update
            policy[s, max_action_idx] = 1
            policy[s, np.arange(len(env.action_space)) != max_action_idx] = 0

            # value update, for deterministic policy, only 0 / 1, v[s] = max(q_v), otherwise, should use integral.
            state_values[s] = max(q_value)

            # compare `max state value` to `original state value`, convergence?
            diff = max(diff, abs(v - state_values[s]))

        print(f'Iteration [{iter_cnts}], diff: {diff}')

        # all state already convergence
        if diff < theta:
            break

    return state_values, policy
