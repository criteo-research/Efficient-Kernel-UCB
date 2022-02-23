import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)

from src.loader import get_agent_by_name, get_env_by_name, get_kernel_by_name

import numpy as np
import time
import os
from datetime import date
today = date.today()
from tqdm import tqdm
import argparse

def save_result(settings, horizon, average_reward, regret, total_time):
    task_name = 'algo:{}'.format(settings['agent'])
    task_name += '|{}:{}'.format('mu', settings['mu'])
    task_name += '|{}:{}'.format('lambda', settings['reg_lambda'])
    task_name += '|{}:{}'.format('C', settings['C'])
    task_name += '|{}:{}'.format('beta', settings['beta'])
    task_name += '|{}:{}'.format('rd', settings['random_seed'])
    task_name += '|{}:{}'.format('kernel', settings['kernel'])
    task_name += '|{}:{}'.format('horizon', horizon)
    task_name += '|{}:{}'.format('env', settings['env'])

    metrics_information = 'average_reward:{}'.format(average_reward)
    metrics_information += '|regret:{}'.format(regret)
    metrics_information += '|total_time:{}'.format(total_time)

    result = '{} {}\n'.format(task_name, metrics_information)
    results_dir = 'results/{}/{}/{}'.format(settings['env'], settings['expname'], today.strftime("%d-%m-%Y"))

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    fname = os.path.join(results_dir, 'metrics.txt')

    with open(fname, 'a') as file:
        file.write(result)

def instantiate_metrics():
    return {
        'time': [],
        'average_reward': [],
        'regret': [],
    }

def do_single_experiment(settings):

    print('Env: {}'.format(settings['env']))
    print('Running experiment with agent {}, lbd {}, mu {}, beta {}, C {}, rd {}'.format(settings['agent'],
                                                                                          settings['reg_lambda'],
                                                                                          settings['mu'],
                                                                                          settings['beta'],
                                                                                          settings['C'],
                                                                                          settings['random_seed']))
    env = get_env_by_name(settings)(settings['random_seed'])
    kernel = get_kernel_by_name(settings)(settings)
    agent = get_agent_by_name(settings)(settings, kernel)
    agent.instantiate(env)
    metrics = instantiate_metrics()
    best_strategy_rewards = []

    if env.horizon:
        settings['T'] = env.horizon

    t0 = time.time()

    for step in tqdm(range(settings['T'] + 1)):

        # choose a random context.
        context, label = env.sample_data()
        # iterate learning algorithm for 1 round.
        action = agent.sample_action(context)
        state = agent.get_state(context, action)
        reward = env.sample_reward_noisy(state, label)[0]
        agent.update_agent(context, action, reward)
        # get best_strategy's reward for the current context.
        best_strategy_rewards.append(env.get_best_reward_in_context(context, label))

        if step % 100 == 0 and step!=0:
            t = time.time() - t0
            metrics['time'].append(t)
            average_reward = np.mean(agent.rewards[1:])
            metrics['average_reward'].append(average_reward)
            sum_best = np.sum(np.array(best_strategy_rewards))
            sum_agent = np.sum(np.array(agent.rewards[1:]))
            regret = sum_best - sum_agent
            save_result(settings, step, average_reward, regret, t)
            print('Average reward: {}'.format(average_reward))
            print('Regret: {}'.format(regret))
            print('Dictionary size: {}'.format(agent.dictionary_size()))

    return metrics

def experiment(args):

    for rd in range(3):
        settings = {
            'agent': args.algo,
            'T': args.max_horizon,
            'random_seed': rd,
            'mu': args.mu,
            'reg_lambda': args.lbd,
            'projection': 'kors',
            'eps': 0.5,
            'beta': args.beta,
            'C': args.C,
            'kernel': args.kernel,
            'env': args.env,
            'expname': args.expname
        }
        do_single_experiment(settings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run scripts for the evaluation of methods')
    parser.add_argument('--algo', nargs="?", default='k_ucb',  choices=['k_ucb', 'ek_ucb', 'cbbkb', 'cbkb'],
                        help='algo method')
    parser.add_argument('--mu', nargs="?", type=float, default=1, help='Projection parameter')
    parser.add_argument('--lbd', nargs="?", type=float, default=1, help='Regularization parameter')
    parser.add_argument('--max_horizon', nargs="?", type=int, default=1000, help='Maximum horizon')
    parser.add_argument('--C', nargs="?", type=float, default=3, help='CBBKB parameter')
    parser.add_argument('--beta', nargs="?", type=float, default=1, help='sampling beta')
    parser.add_argument('--kernel', nargs="?", default='gauss',  choices=['gauss', 'exp'],
                        help='kernel choice')
    parser.add_argument('--env', nargs="?", default='squares',  choices=['bump', 'squares', 'step_diag'],
                        help='environment')
    parser.add_argument('--expname', nargs="?", type=str, default='experiment', help='name of the experiment')
    experiment(parser.parse_args())
