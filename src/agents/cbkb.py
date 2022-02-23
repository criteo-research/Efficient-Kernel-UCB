import jax.scipy as jsp
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
from jax import grad, hessian
from jax import jit
from functools import partial
import numpy as np
#
import os
import sys

# base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
# sys.path.append(base_dir)

base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)

from src.agents.ek_ucb import EK_UCB
from src.utils import sherman_morrison_update, schur_first_order_update

EPS = 1e-6


class CBKB(EK_UCB):

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        super(CBKB, self).__init__(*args)

    def instantiate(self, env):
        self.action_anchors = env.get_anchor_points()
        self.a_dimension = env.action_dimension
        self.x_dimension = env.context_dimension
        context, label = env.sample_data()
        idx = self.rng.choice(self.action_anchors.shape[0])
        action = np.array([self.action_anchors[idx]])
        state = self.get_state(context, action)
        reward = env.sample_reward_noisy(state, label)[0]
        self.past_states = jnp.array(self.get_state(context, action))
        self.rewards = jnp.array([reward])
        self.projection_dictionary = jnp.array(self.get_state(context, action))

        self.K_ZZ = self.kernel.gram_matrix(self.projection_dictionary)
        self.K_ZZ += EPS * jnp.eye(self.K_ZZ.shape[0])
        self.inv_K_ZZ = jnp.linalg.inv(self.K_ZZ)

        state = self.get_state(context, action)
        K_Z_s = self.kernel.evaluate(self.projection_dictionary, state)

        self.Lambda = jnp.linalg.inv(np.dot(K_Z_s, K_Z_s.T) + self.reg_lambda * self.K_ZZ)
        self.Gamma = jnp.array(reward) * K_Z_s

    def get_updated_dictionary(self):

        dico = jnp.array([self.past_states[0]])
        q = 10 * self.settings['C']
        for state in self.past_states:
            state = state.reshape((1, -1))
            tau = self.bkb_score(state, self.projection_dictionary, self.Lambda, self.Gamma, self.inv_K_ZZ)
            p = max(min(q * tau, 1), 0)
            p = jnp.array(p).reshape(1)
            z = self.rng.binomial(1, p)
            if z:
                dico = jnp.concatenate([dico, state])
        return dico

    def update_agent(self, context, action, reward):
        state = self.get_state(context, action)
        self.projection_dictionary = self.get_updated_dictionary()
        past_states, past_rewards = self.get_story_data()

        self.K_ZZ = self.kernel.gram_matrix(self.projection_dictionary)
        self.K_ZZ += EPS * jnp.eye(self.K_ZZ.shape[0])
        self.inv_K_ZZ = jnp.linalg.inv(self.K_ZZ)

        state = self.get_state(context, action)
        S = jnp.concatenate([past_states, state])
        K_Z_S = self.kernel.evaluate(self.projection_dictionary, S)

        self.Lambda = jnp.linalg.inv(np.dot(K_Z_S, K_Z_S.T) + self.reg_lambda * self.K_ZZ)

        Y_S = jnp.concatenate([past_rewards, jnp.array([reward])])
        self.Gamma = jnp.dot(K_Z_S, Y_S)

        self.update_data_pool(context, action, reward)

    def bkb_score(self, state, Z, Lambda, Gamma, inv_K_ZZ):

        # Virtual update
        K_Z_s = self.kernel.evaluate(Z, state)
        K_ss = self.kernel.evaluate(state, state)
        first_term = jnp.dot(K_Z_s.T, jnp.dot(Lambda, K_Z_s))
        tau = first_term + (1 / self.reg_lambda) * (K_ss - jnp.dot(K_Z_s.T, jnp.dot(inv_K_ZZ, K_Z_s)))
        return tau
