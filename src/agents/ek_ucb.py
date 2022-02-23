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

# base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
# sys.path.append(base_dir)

base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)

from src.agents.base import KernelUCB
from src.utils import sherman_morrison_update, schur_first_order_update, schur_first_order_update_fast


EPS = 0

from profilehooks import timecall

class EK_UCB(KernelUCB):

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        super(EK_UCB, self).__init__(*args)
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

        # self.projection_dictionary = self.get_fixed_dictionary()
        self.projection_dictionary = jnp.array(self.get_state(context, action))

        K_ZZ = self.kernel.gram_matrix(self.projection_dictionary)
        K_ZZ += EPS * jnp.eye(K_ZZ.shape[0])
        self.inv_K_ZZ = jnp.linalg.inv(K_ZZ)

        state = self.get_state(context, action)
        K_Z_s = self.kernel.evaluate(self.projection_dictionary, state)
        self.K_ZS = K_Z_s.T.tolist()

        self.Lambda = np.linalg.inv(np.dot(K_Z_s, K_Z_s.T) + self.reg_lambda * K_ZZ)
        self.Gamma = jnp.array(reward) * K_Z_s

        self.proba_sampling_anchors = np.ones(self.projection_dictionary.shape[0])
        self.inv_kors_matrix = jnp.linalg.inv(K_ZZ + self.settings['mu'] * jnp.eye(K_ZZ.shape[0]))

        self.a_dimension = env.action_dimension
        self.x_dimension = env.context_dimension

    def get_init_dictionary(self,context,action):
        projection_dictionary = jnp.array([context, action]).T
        return projection_dictionary

    def get_fixed_dictionary(self):
        initial_size = 2
        action_anchors = self.rng.choice(np.arange(0, 1, 0.05), size=initial_size*self.a_dimension, replace=False).reshape((initial_size, self.a_dimension))
        context_anchors = self.rng.choice(np.arange(0, 1, 0.05), size=initial_size*self.x_dimension, replace=False).reshape((initial_size, self.x_dimension))
        projection_dictionary = jnp.concatenate([context_anchors, action_anchors], axis=1)

        return projection_dictionary

    def dictionary_size(self):
        return self.projection_dictionary.shape[0]

    # @timecall(immediate=False)
    def get_updated_dictionary(self, state):
        if self.settings['projection'] == 'fixed':
            self.projection_dictionary = self.get_fixed_dictionary()
            return True
        elif self.settings['projection'] == 'kors':
            mu = self.settings['mu']
            eps = self.settings['eps']
            gamma = self.settings['beta'] * self.settings['reg_lambda']
            Z = self.projection_dictionary
            A = self.inv_kors_matrix
            sqprob = np.expand_dims(np.sqrt(self.proba_sampling_anchors),axis=1)

            self.K_Zs = self.kernel.evaluate(Z, state)
            b = self.K_Zs / sqprob
            self.kss = self.kernel.evaluate(state,state)
            z = A.dot(b)
            u = b.T.dot(z)
            s = 1 / (self.kss + mu - u)
            tau = ((1+eps)/mu) * ( u + s * ( u - self.kss) * (u - self.kss))
            p = max(min(gamma * tau, 1), 0)
            p = jnp.array(p).reshape(1)
            update = self.rng.binomial(1, p)
            if update:
                self.proba_sampling_anchors = jnp.concatenate([self.proba_sampling_anchors, p])
                self.inv_kors_matrix = schur_first_order_update_fast(self.inv_kors_matrix, z /np.sqrt(p), b/np.sqrt(p), self.kss/p + mu)
                self.projection_dictionary = jnp.concatenate([Z, state])
                return True
            else:
                return False
        else:
            print('Not Implemented!')


    # @timecall(immediate=False)
    def update_agent(self, context, action, reward):
        state = self.get_state(context, action)
        is_dict_updated = self.get_updated_dictionary(state)
        past_states, past_rewards = self.get_story_data()

        # Projection dictionary changes
        if is_dict_updated:
            self.efficient_update_projected_matrices_and_dico(
                past_states, state, past_rewards, reward)

        # Projection dictionary does not change
        else:
            self.efficient_update_projected_matrices(reward)
        self.update_data_pool(context, action, reward)


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

    # @timecall(immediate=False)
    def efficient_update_projected_matrices_and_dico(self, past_states, state,
                                                     past_rewards, reward):
        self.K_ZS.append((self.K_Zs.reshape(-1)))
        Z = self.projection_dictionary
        S = jnp.concatenate([past_states, state])
        K_sS = self.kernel.evaluate(S, state)
        K_ZS = np.array(self.K_ZS)
        K_ZS = np.concatenate((K_ZS, K_sS),axis=1)
        self.K_ZS = K_ZS.tolist()

        ### Gamma update
        # State update to previous anchor points
        self.Gamma += reward[0] * self.K_Zs

        # Block update to new anchor point, for all states
        Y_S = jnp.concatenate([past_rewards, jnp.array([reward])])
        slack = K_sS.T.dot(Y_S)
        self.Gamma = jnp.block([[self.Gamma], [slack]])

        ### Lambda update
        # State update to previous anchor points
        Lambda_Z = sherman_morrison_update(self.Lambda, self.K_Zs, self.K_Zs)
        a = K_ZS.T.dot(K_sS)
        c = self.reg_lambda * self.kss + a[-1]
        b = self.reg_lambda * self.K_Zs + a[0:-1] 
        self.Lambda = schur_first_order_update(Lambda_Z, b, c)

        ### inv_K_ZZ update
        self.inv_K_ZZ = schur_first_order_update(self.inv_K_ZZ, self.K_Zs, self.kss)

    # @timecall(immediate=False)
    def efficient_update_projected_matrices(self, reward):
        self.Lambda = sherman_morrison_update(self.Lambda, self.K_Zs, self.K_Zs)
        self.Gamma += reward * self.K_Zs
        self.K_ZS.append(np.array(self.K_Zs.reshape(-1)))

    # @timecall(immediate=False)
    def recompute_inverses(self):
        past_states, past_rewards = self.get_story_data()
        Z = self.projection_dictionary
        K_ZS = np.array(self.K_ZS)
        K_ZZ = self.kernel.evaluate(Z, Z)
        self.Lambda = np.linalg.inv(np.dot(K_ZS.T, K_ZS) + self.reg_lambda*K_ZZ)
        self.inv_K_ZZ = np.linalg.inv(K_ZZ + EPS * np.eye(K_ZZ.shape[0]))
        sqprob = np.expand_dims(np.sqrt(self.proba_sampling_anchors),axis=1)
        self.inv_kors_matrix =np.linalg.inv(sqprob * K_ZZ * sqprob.T + self.settings['mu']*np.eye(K_ZZ.shape[0]))

