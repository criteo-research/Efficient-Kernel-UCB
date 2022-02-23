import numpy as np
from kernel import Kernel

# this env is piecewise constant valued, with values in {0, a=0.5, b=1} it is null except on upper and lower diagonal of a=-x
# an exemple with env.discr=4 is display:
#  ---------------
# | 0 | a | 0 | 0 |
# | b | 0 | b | 0 |
# | 0 | a | 0 | a |
# | 0 | 0 | b | 0 |
#  ---------------

actions = np.arange(0, 1, 0.2)
contexts = np.arange(0, 1, 0.2)

class Squares:

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

    context, action = state[:,:self.context_dimension], state[:, self.context_dimension:]
    x, y = context, action
    b_cond = ((0 <= x) & (x < 0.25) & (0.25 <= y) & (y < 0.5)) | (
                (0.5 <= x) & (x < 0.75) & (0.25 <= y) & (y < 0.5)) | ((0.5 <= x) & (x < 0.75) & (0.75 <= y) & (y <= 1))
    a_cond = ((0.25 <= x) & (x < 0.5) & (0 <= y) & (y < 0.25)) | (
                (0.25 <= x) & (x < 0.5) & (0.5 <= y) & (y < 0.75)) | ((0.75 <= x) & (x <= 1) & (0.5 <= y) & (y < 0.75))
    r = np.where(a_cond, self.a, (np.where(b_cond, self.b, 0))).squeeze(1)
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
