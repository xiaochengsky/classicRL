# -*- coding: utf-8 -*-
# @Time: 2025/2/20 下午4:38
# @Author: YANG.C
# @File: main.py

from algs import *
from src.grid_world import GridWorld
from example.arguments import args

if __name__ == '__main__':
    env = GridWorld()
    env.reset()

    if args.alg == 'value_iter':
        alg = value_iter
    elif args.alg == 'policy_iter':
        alg = policy_iter
    elif args.alg == 'monte_carlo_basic':
        alg = menton_calro_basic
    elif args.alg == 'monte_carlo_explore':
        alg = monte_carlo_exploring
    elif args.alg == 'menton_carlo_e_greedy':
        alg = menton_carlo_e_greedy
    elif args.alg == 'sarsa':
        alg = sarsa
    elif args.alg == 'sarsa_n_step':
        alg = sarsa_n_step
    elif args.alg == 'q_learning_on_policy':
        alg = q_learning_on_policy
    elif args.alg == 'dqn':
        alg = dqn
    else:
        raise ValueError("cannot find this alg!")

    if args.alg != 'dqn':
        optimal_values, policy = alg(env)
    else:
        optimal_values, _ = value_iter(env)
        optimal_values, policy = alg(env, optimal_values)

    env.render()
    env.add_policy(policy)
    env.add_state_values(optimal_values)
    env.render(animation_interval=50)
