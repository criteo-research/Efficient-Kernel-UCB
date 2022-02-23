import os
import sys
import jax.scipy as jsp
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)

from src.agents.k_ucb import K_UCB
from src.agents.ek_ucb import EK_UCB
from src.agents.cbkb import CBKB
from src.agents.cbbkb import CBBKB
from src.agents.supk_ucb import SupK_UCB

from src.env.bump import Bump
from src.env.squares import Squares
from src.env.step_diag import StepDiag

def get_agent_by_name(settings):
    if settings['agent'] == 'k_ucb':
        return K_UCB
    elif settings['agent'] == 'ek_ucb':
        return EK_UCB
    elif settings['agent'] == 'cbkb':
        return CBKB
    elif settings['agent'] == 'cbbkb':
        return CBBKB
    elif settings['agent'] == 'supk_ucb':
        return SupK_UCB
    else:
        raise NotImplementedError

def get_env_by_name(settings):
    if settings['env'] == 'bump':
        return Bump
    elif settings['env'] == 'squares':
        return Squares
    elif settings['env'] == 'step_diag':
        return StepDiag
    else:
        raise NotImplementedError

from src.kernel import Gaussian, Exponential

def get_kernel_by_name(settings):
    if settings['kernel'] == 'gauss':
        return Gaussian
    elif settings['kernel'] == 'exp':
        return Exponential
    else:
        raise NotImplementedError
