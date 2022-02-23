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

from src.agents.base import KernelUCB

class SupK_UCB(KernelUCB):

  def __init__(self, *args):
    """Initializes the class

    Attributes:
        random_seed (int):  random seed for data generation process

    """
    super().__init__(*args)
    self.round = 1

  def instantiate(self, env):
    self.action_anchors = env.get_anchor_points()
    context, label = env.sample_data()
    self.N = self.action_anchors.shape[0]
    idx = self.rng.choice(self.N)
    action = np.array([self.action_anchors[idx]])
    state = self.get_state(context, action)
    reward = env.sample_reward_noisy(state, label)[0]
    self.past_states = np.array(self.get_state(context, action))
    self.rewards = np.array([reward])
    self.sets = np.zeros((self.N, 1), dtype=int)

  def update_agent(self, context, action, reward):
    self.update_data_pool(context, action, reward)

  def basekernel_ucb(self, index_set, state):
    eps = 1e-6
    index_set = index_set.astype(bool)
    S = self.past_states[index_set]
    Y = self.rewards[index_set]
    K_S = self.kernel.evaluate(S, S)
    K_S_s = self.kernel.evaluate(S, state)
    K_matrix_inverse = np.linalg.inv(K_S + eps * np.eye(K_S.shape[0]))
    mean = np.dot(K_S_s.T, np.dot(K_matrix_inverse, Y))
    K_ss = self.kernel.evaluate(state, state)
    std = (1 / self.reg_lambda) * (K_ss - np.dot(K_S_s.T, np.dot(K_matrix_inverse, K_S_s)))
    return np.squeeze(mean), np.squeeze(np.sqrt(std))

  def sample_action(self, context):
    self.set_beta()
    action_set = self.action_anchors
    eta = self.beta_t * self.settings['beta']
    s = 1

    while True:
      zipped = [self.basekernel_ucb(self.sets[s], self.get_state(context, action)) for action in action_set]
      unzipped_object = zip(*zipped)
      means, stds = list(unzipped_object)
      means = np.stack(list(means))
      stds = np.stack(list(stds))
      condition = eta * np.array(stds) > 1 / np.sqrt(self.settings['T'])
      variance_condition = not np.sum(condition) >= 1
      if variance_condition:
        idx = np.argmax(np.array(means + eta * stds))
        action = action_set[idx]
        self.sets = np.hstack([self.sets, np.zeros((self.N, 1), dtype=int)])
        break
      else:
        condition = eta * np.array(stds) > 1 / 2 ** s
        variance_condition = not np.sum(condition) >= 1
        if variance_condition:
          index_condition = (means + eta * stds >= np.max(means + eta * stds) - 2 ** (1 - s))
          action_set = action_set[index_condition]
          s += 1
        else:
          condition = eta * np.array(stds) > 1 / 2 ** s
          indexes = np.arange(action_set.shape[0])
          idx_action = np.random.choice(indexes[condition])
          action = action_set[idx_action]
          self.sets = np.hstack([self.sets, np.zeros((self.N, 1), dtype=int)])
          self.sets[idx_action, -1] = 1
          break

    self.round += 1
    return action