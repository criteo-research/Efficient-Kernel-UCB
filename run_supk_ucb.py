import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)

import argparse
from datetime import date
today = date.today()
from run import do_single_experiment

def experiment(args):

    lbds = [1, 10, 100]
    betas = [1, 10]
    count = 0
    mu = 0
    c = 1

    for lbd in lbds:
        for beta in betas:
            for rd in range(3):
                settings = {
                    'agent': 'supk_ucb',
                    'T': args.T,
                    'random_seed': rd,
                    'mu': mu,
                    'reg_lambda': lbd,
                    'projection': 'kors',
                    'eps': 0.5,
                    'beta': beta,
                    'C': c,
                    'kernel': 'gauss',
                    'env': args.env,
                    'expname': 'experiment_supk_ucb'
                }
                do_single_experiment(settings)
                count += 1
    print('Done {} experiments'.format(count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run scripts for the evaluation of methods')
    parser.add_argument('--env', nargs="?", type=str, default='bump', choices=['bump', 'step_diag', 'squares'], help='Environment choice')
    parser.add_argument('--T', nargs="?", type=int, default=1000, help='Max horizon')
    experiment(parser.parse_args())
