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

    mus = [1, 10, 100]
    lbds = [1, 10, 100]
    betas = [0.001, 0.02, 0.05, 0.1]
    count = 0
    c = 1

    for lb in lbds:
        condition = (lb == 10) and (args.env == 'bump')
        if condition :
            for mu in mus:
                for beta in betas:
                    for rd in range(3):
                        settings = {
                            'agent': 'ek_ucb',
                            'T': args.T,
                            'random_seed': rd,
                            'mu': mu,
                            'reg_lambda': lb,
                            'projection': 'kors',
                            'eps': 0.5,
                            'beta': beta,
                            'C': c,
                            'kernel': 'gauss',
                            'env': args.env,
                            'expname': 'experiment_ek_ucb'
                        }
                        do_single_experiment(settings)
                        count += 1
        else:
            mu = lb
            for beta in betas:
                for rd in range(3):
                    settings = {
                        'agent': 'ek_ucb',
                        'T': args.T,
                        'random_seed': rd,
                        'mu': mu,
                        'reg_lambda': lb,
                        'projection': 'kors',
                        'eps': 0.5,
                        'beta': beta,
                        'C': c,
                        'kernel': 'gauss',
                        'env': args.env,
                        'expname': 'experiment_ek_ucb'
                    }
                    do_single_experiment(settings)
                    count += 1

    print('Done {} experiments'.format(count))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run scripts for the evaluation of methods')
    parser.add_argument('--env', nargs="?", type=str, default='bump', choices=['bump', 'step_diag', 'squares'], help='Environment choice')
    parser.add_argument('--T', nargs="?", type=int, default=1000, help='Max horizon')
    experiment(parser.parse_args())
