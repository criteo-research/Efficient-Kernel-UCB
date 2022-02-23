import jax.scipy as jsp
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
from jax import grad, hessian
from jax import jit
from functools import partial
import numpy as np
from profilehooks import timecall


def instantiate_data_pool():
    return {
        'contexts': [],
        'actions': [],
        'rewards': []
    }

class KernelUCB:

    def __init__(self, settings, kernel):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        self.rng = np.random.RandomState(123)
        self.reg_lambda = settings['reg_lambda']
        self.kernel = kernel
        self.settings = settings

    def dictionary_size(self):
        return 0

    def get_story_data(self):
      return self.past_states, self.rewards

    def set_gram_matrix(self):
        K = self.kernel.gram_matrix(self.past_states)
        K += self.reg_lambda * jnp.eye(K.shape[0])
        self.K_matrix_inverse = jnp.linalg.inv(K)

    def set_beta(self):
        self.beta_t = 0.1

    def get_upper_confidence_bound(self, state, K_matrix_inverse, S, rewards):
        K_S_s = self.kernel.evaluate(S, state)
        mean = jnp.dot(K_S_s.T, jnp.dot(K_matrix_inverse, rewards))
        K_ss = self.kernel.evaluate(state, state)
        std = (1/self.reg_lambda)*(K_ss - jnp.dot(K_S_s.T, jnp.dot(K_matrix_inverse, K_S_s)))
        ucb = mean + self.beta_t * jnp.sqrt(std)
        return jnp.squeeze(ucb)

    def instantiate(self, env):
        self.action_anchors = env.get_anchor_points()
        context, label = env.sample_data()
        idx = self.rng.choice(self.action_anchors.shape[0])
        action = np.array([self.action_anchors[idx]])
        state = self.get_state(context, action)
        reward = env.sample_reward_noisy(state, label)[0]
        self.past_states = jnp.array(self.get_state(context, action))
        self.rewards = jnp.array([reward])
        self.set_gram_matrix()

    @timecall(immediate=False)
    def sample_action(self, context):
        self.set_beta()
        S, rewards = self.get_story_data()
        args = self.K_matrix_inverse, S, rewards
        return self.continuous_inference(context, args)

    def dicionary_size(self):
        return 0

    def continuous_inference(self, context, args):
        nb_gradient_steps = 0

        if nb_gradient_steps == 0:
            return self.discrete_inference(context,args)
        else:
            def func(action):
              state = self.get_state(context, action)
              return self.get_upper_confidence_bound(state, *args)

            a0 = self.discrete_inference(context, args)
            max_hessian_eigenvalue = jnp.max(jsp.linalg.eigh(hessian(func)(a0), eigvals_only=True))
            step_size = jnp.nan_to_num(1 / max_hessian_eigenvalue)
            a_t = a0
            for _ in range(nb_gradient_steps):
                gradient = jnp.nan_to_num(grad(func)(a_t))
                a_t -= step_size * gradient
            return a_t

    def get_state(self, context, action):
      context, action = context.reshape((1, -1)), action.reshape((1, -1))
      return jnp.concatenate([context, action], axis=1)

    def get_batch_ucb(self, context, grid, args):
        return jnp.array([self.get_upper_confidence_bound(self.get_state(context, a), *args) for a in grid])

    def discrete_inference(self, context, args):
        grid = self.action_anchors
        ucb_all_actions = self.get_batch_ucb(context,grid,args) 
        idx = jnp.argmax(ucb_all_actions)
        grid = jnp.array(grid)
        return jnp.array([grid[idx]])

    def update_agent(self, context, action, reward):
        self.update_data_pool(context, action, reward)
        self.set_gram_matrix()

    def update_data_pool(self, context, action, reward):
        state = self.get_state(context, action)
        self.past_states = jnp.concatenate([self.past_states, state])
        self.rewards = jnp.concatenate([self.rewards, jnp.array([reward])])

