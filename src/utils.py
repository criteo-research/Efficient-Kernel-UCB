import os
import sys
import jax.scipy as jsp
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
from jax import grad, hessian
from jax import jit
from functools import partial
import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)
#

def sherman_morrison_update(A_inv, u, v):
      """"
      update of (A+ u.v^\top)^{-1} using A^{-1}
      params
      A_inv: np.array of size (n,n)
      u: np.array of size (n, 1)
      v = np.array of size (n, 1)
      """
      bb = np.dot(A_inv, u)
      d = np.dot(v.T, bb)
      D = np.dot(bb, np.dot(v.T, A_inv))
      return A_inv - 1/(1+d)*D

def schur_first_order_update(A_inv, b, c):
      """"
      update of the inverse of A_{t+1} = [[A_t, b], [b^T, c]]
      using the inverse of A_{t}
      params
      A_inv: np.array of size (n,n)
      u: np.array of size (n, 1)
      v = np.array of size (n, 1)
      """
      z = np.dot(A_inv, b)
      s = 1/(c - np.dot(z.T, b))
      Z_12 = - s * z
      Z_11 = A_inv + s * np.dot(z, z.T)
      return np.block([[Z_11, Z_12], [Z_12.T, s]])

def schur_first_order_update_fast(A_inv, A_inv_b, b, c):
      """"
      update of the inverse of A_{t+1} = [[A_t, b], [b^T, c]]
      using the inverse of A_{t}
      params
      A_inv: np.array of size (n,n)
      u: np.array of size (n, 1)
      v = np.array of size (n, 1)
      """
      z = A_inv_b 
      s = 1/(c - np.dot(z.T, b))
      Z_12 = - s * z
      Z_11 = A_inv + s * np.dot(z, z.T)
      return np.block([[Z_11, Z_12], [Z_12.T, s]])

