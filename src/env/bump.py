import numpy as np

actions = np.arange(0, 1, 0.2)
contexts = np.arange(0, 1, 0.2)

class Bump:

  def __init__(self, rd, actions=actions, contexts=contexts):
    self.input_mesh = np.array(np.meshgrid(actions, contexts))
    self.context_numbers = contexts.size
    self.actions = actions
    self.contexts = contexts
    self.action_dimension = 1
    self.context_dimension = 5
    self.scale = 1
    self.sampling_rng = np.random.RandomState(rd)
    self.fixed_env_rd = np.random.RandomState(123)
    self._set_anchor_points()
    idx  = self.fixed_env_rd.choice(self.anchors.shape[0])
    self.a_star = self.anchors[idx]
    self.x_star = self.fixed_env_rd.uniform(0, 1, size=(1, self.context_dimension))
    self.w_star = self.fixed_env_rd.uniform(-1, 1, size=self.context_dimension)
    self.horizon = None


  def sample_reward(self, state, label):
    context, action = state[:,:self.context_dimension], state[:, self.context_dimension:]
    term = np.linalg.norm(action - self.a_star, ord=1) + np.dot(context - self.x_star, self.w_star).squeeze()
    r = max(0, 1 - term)
    return np.array([r])

  def sample_reward_noisy(self, state, label):
    return [self.sample_reward(state, label) + self.sampling_rng.normal(loc=0.0, scale=0.1)]

  def find_best_input_in_joint_space(self, joint_pair, label):
    return np.argmax(self.sample_reward(joint_pair, label))

  def get_best_reward_in_context(self, context, label):
    term = np.dot(context - self.x_star, self.w_star).squeeze()
    r = max(0, 1 - term)
    return r + self.sampling_rng.normal(loc=0.0, scale=0.1)

  def sample_data(self):
    return np.array(self.sampling_rng.uniform(low=0, high=1, size=(1, self.context_dimension)), dtype=np.float64), None

  def _set_anchor_points(self):
    initial_size = 10
    self.anchors = self.fixed_env_rd.choice(np.arange(0, 1, 0.01), size=initial_size*self.action_dimension,
                                     replace=False).reshape((initial_size, self.action_dimension))

  def get_anchor_points(self):
    return self.anchors
