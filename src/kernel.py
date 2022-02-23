import jax.scipy as jsp
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
from jax import grad, hessian
from jax import jit
import jax as jax
from functools import partial
import numpy as np

@jax.jit
def sqeuclidean_distance(x, y):
    return jnp.sum((x-y)**2)

# RBF Kernel
@jax.jit
def rbf_kernel(gamma, x, y):
    return jnp.exp( - gamma * sqeuclidean_distance(x, y))

# Exponential Kernel
@jax.jit
def exp_kernel(gamma, x, y):
    return jnp.exp( - gamma * jnp.sqrt(sqeuclidean_distance(x, y)))

def gram(func, params, x, y):
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(params, x1, y1))(y))(x)

class Kernel:

    def __init__(self, settings):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        self.context_dimension = 1
        self._param = 0.1

    def gram_matrix(self, states):
        return self._pairwise(states, states)

    def evaluate(self, state1, state2):
        return self._pairwise(state1, state2)

    def _pairwise(self, X1, X2):
        pass


class Gaussian(Kernel):

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        super(Gaussian, self).__init__(*args)
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        self._std = self._param

    def _pairwise(self, X1, X2):
        """
        Args:
            X1 (np.ndarray)
            X2 (np.ndarray)
        """
        return gram(rbf_kernel, 1/(2* self._std ** 2),X1,X2)

class Exponential(Kernel):

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        super(Exponential, self).__init__(*args)
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        self._alpha = 10

    def _pairwise(self, X1, X2):
        """
        Args:
            X1 (np.ndarray)
            X2 (np.ndarray)
        """
        return gram(exp_kernel, self._alpha,X1,X2)