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


base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)

from src.agents.ek_ucb import EK_UCB
from src.utils import sherman_morrison_update, schur_first_order_update

EPS = 0.

class CBBKB(EK_UCB):

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        super(CBBKB, self).__init__(*args)
        self.K_Zs = 0
        self.kss = 0
        self.k_ZS = 0

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
        self.K_ZS = K_Z_s.T.tolist()

        self.Lambda = jnp.linalg.inv(np.dot(K_Z_s, K_Z_s.T) + self.reg_lambda * self.K_ZZ)
        self.Gamma = jnp.array(reward) * K_Z_s
        self.round_counter = 1
        self.last_batch_index = 0
        self.bkb_scores = []

    def get_updated_dictionary(self):
        A = self.Lambda - self.inv_K_ZZ / self.reg_lambda
        dico = [self.past_states[0]]
        q = 10 * self.settings['C']
        for state in self.past_states[1:]:
            state = state.reshape((1, -1))
            tau = self.bkb_score(state.reshape((1, -1)), self.projection_dictionary, A)
            p = max(min(q * tau, 1), 0)
            p = jnp.array(p).reshape(1)
            update = self.rng.binomial(1, p)
            if update:
                flat_state = state.reshape(-1)
                dico.append(flat_state)

        return jnp.array(dico)

    def dictionary_size(self):
        return super().dictionary_size()

    def get_update_condition_score(self, state):
        # states = self.past_states[self.last_batch_index:]
        A = self.Lambda - self.inv_K_ZZ / self.reg_lambda
        score = self.bkb_score(state.reshape((1, -1)), self.projection_dictionary, A)
        self.bkb_scores.append(score)
        return 1 + np.sum(self.bkb_scores)


    def update_agent(self, context, action, reward):
        state = self.get_state(context, action)
        condition = (self.get_update_condition_score(state) > self.settings['C'])

        if condition:
            self.projection_dictionary = self.get_updated_dictionary()
            self.update_restart_projected_matrices(state, reward,
                                                   self.projection_dictionary)
            self.last_batch_index = self.round_counter
            self.bkb_scores = []

        # Projection dictionary does not change
        else:
            self.efficient_update_projected_matrices(reward)

        self.update_data_pool(context, action, reward)
        self.round_counter += 1
        # if len(past_rewards) % 100 == 0:
        #     self.recompute_inverses()

    def update_restart_projected_matrices(self, state, reward, Z):
        past_states, past_rewards = self.get_story_data()
        K_ZZ = self.kernel.gram_matrix(Z)
        K_ZZ += EPS * jnp.eye(K_ZZ.shape[0])
        self.inv_K_ZZ = jnp.linalg.inv(K_ZZ)

        S = jnp.concatenate([past_states, state])
        K_ZS = self.kernel.evaluate(Z, S)

        self.Lambda = jnp.linalg.inv(np.dot(K_ZS, K_ZS.T) + self.reg_lambda * K_ZZ)

        Y_S = jnp.concatenate([past_rewards, jnp.array([reward])])
        self.K_Zs = K_ZS
        self.Gamma = jnp.dot(K_ZS, Y_S)

    def bkb_score(self, state, Z, A):
        # Virtual update
        self.K_Zs = self.kernel.evaluate(Z, state)
        K_ss = self.kernel.evaluate(state, state)
        tau = (1/self.reg_lambda)*K_ss + jnp.dot(self.K_Zs.T, jnp.dot(A, self.K_Zs))
        return tau

    def efficient_update_projected_matrices(self, reward):
        self.Lambda = sherman_morrison_update(self.Lambda, self.K_Zs, self.K_Zs)
        self.Gamma += reward * self.K_Zs
        self.K_ZS.append(np.array(self.K_Zs.reshape(-1)))

    # @timecall(immediate=False)
    def sample_action(self, context):
        self.set_beta()
        args = self.Lambda.dot(self.Gamma), self.Lambda - self.inv_K_ZZ/self.reg_lambda, self.projection_dictionary
        return self.continuous_inference(context, args)

    # @timecall(immediate=False)
    def get_upper_confidence_bound(self, state, LambdaGamma, A , Z):
        K_Z_s = self.kernel.evaluate(Z, state)
        K_ss = self.kernel.evaluate(state, state)
        mean = jnp.dot(K_Z_s.T, LambdaGamma)
        std = (1/self.reg_lambda)*K_ss + jnp.dot(K_Z_s.T, jnp.dot(A, K_Z_s))
        ucb = mean + self.beta_t * jnp.sqrt(std)
        return jnp.squeeze(ucb)
