import jax.scipy as jsp
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
from jax import grad, hessian
from jax import jit
from functools import partial
import numpy as np

import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)
#
from src.agents.base import KernelUCB
from profilehooks import timecall

class K_UCB(KernelUCB):

  def __init__(self, *args):
    """Initializes the class

    Attributes:
        random_seed (int):  random seed for data generation process

    """
    super(K_UCB, self).__init__(*args)

  def update_agent(self, context, action, reward):
    state = self.get_state(context, action)
    S, _ = self.get_story_data()
    self.K_matrix_inverse = self.efficient_update_gram_matrix(S, state, self.K_matrix_inverse)
    self.update_data_pool(context, action, reward)

  # @partial(jit, static_argnums=(0,))
  @timecall(immediate=False)
  def efficient_update_gram_matrix(self, S, state, K_matrix_inverse):
    K_S_s = self.kernel.evaluate(S, state)
    K_ss = self.kernel.evaluate(state, state)
    s = K_ss + self.reg_lambda - jnp.dot(K_S_s.T, jnp.dot(K_matrix_inverse, K_S_s))
    Z_12 = - 1/s * (jnp.dot(K_matrix_inverse, K_S_s))
    Z_21 =  - 1/s * (jnp.dot(K_S_s.T, K_matrix_inverse))
    Z_11 = K_matrix_inverse + s * jnp.dot(Z_12, Z_21)
    K_matrix_inverse = jnp.block([[Z_11, Z_12], [Z_21, 1/s]])
    return K_matrix_inverse

