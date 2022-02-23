import numpy as np
from kernel import Kernel

# This
# env is diagonal
# piecewise
# constant, i.e.
# if | x-(a+0.15) | < 0.1 then 1
# if | x-(a-0.15) | < 0.1 then 0.5
# else 0

actions = np.arange(0, 1, 0.2)
contexts = np.arange(0, 1, 0.2)

class StepDiag:

  def __init__(self, rd, actions=actions, contexts=contexts):
    self.input_mesh = np.array(np.meshgrid(actions, contexts))
    self.context_numbers = contexts.size
    self.actions = actions
    self.contexts = contexts
    self.action_dimension = 1
    self.context_dimension = 1
    self.scale = 1
    self.sampling_rng = np.random.RandomState(rd)
    self.fixed_env_rd = np.random.RandomState(123)
    self._set_anchor_points()
    idx  = self.fixed_env_rd.choice(self.anchors.shape[0])
    self.a = 1
    self.b = 0.5
    self.horizon = None


  def sample_reward(self, state, label):
    # state = np.array(state, dtype=np.float64)
    # print(state)
    context, action = state[:,:self.context_dimension], state[:, self.context_dimension:]
    b_cond = np.linalg.norm(context - (action-0.15*np.ones(self.context_dimension)), ord=1)<0.1
    a_cond = np.linalg.norm(context - (action+0.15*np.ones(self.context_dimension)), ord=1)<0.1
    r = np.array([np.where(a_cond, self.a, (np.where(b_cond, self.b, 0)))])
    return r

  def sample_reward_noisy(self, state, label):
    return [self.sample_reward(state, label) + self.sampling_rng.normal(loc=0.0, scale=0.1)]

  def find_best_input_in_joint_space(self, joint_pair, label):
    return np.argmax(self.sample_reward(joint_pair, label))

  def get_best_reward_in_context(self, context, label):
    r= self.a
    return r + self.sampling_rng.normal(loc=0.0, scale=0.1)

  def sample_data(self):
    return np.array(self.sampling_rng.uniform(low=0, high=1, size=(1, self.context_dimension)), dtype=np.float64), None

  def _set_anchor_points(self):
    initial_size = 10
    self.anchors = self.fixed_env_rd.choice(np.arange(0, 1, 0.01), size=initial_size*self.action_dimension,
                                     replace=False).reshape((initial_size, self.action_dimension))

  def get_anchor_points(self):
    return self.anchors
