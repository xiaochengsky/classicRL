# -*- coding: utf-8 -*-
# @Time: 2025/2/20 下午4:38
# @Author: YANG.C
# @File: main.py

from algs import value_iter
from src.grid_world import GridWorld
from example.arguments import args

if __name__ == '__main__':
    env = GridWorld()
    env.reset()

    if args.alg == 'value_iter':
        optimal_values, policy = value_iter(env)
    else:
        raise ValueError("cannot find this alg!")

    env.render()
    env.add_policy(policy)
    env.add_state_values(optimal_values)
    env.render(animation_interval=5)
