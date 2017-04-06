import tensorflow as tf
import numpy as np
import sys
import gym
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import fully_connected as fc

flags = tf.app.flags

flags.DEFINE_string("ENVIRONMENT", "CartPole-v0", "environment")
flags.DEFINE_string("AGENT", "REINFORCE", "agent")
flags.DEFINE_float("GAMMA", 0.99, "discount factor")
flags.DEFINE_float("LEARNING_RATE", 0.0001, "learning rate")
flags.DEFINE_integer("TRAINING_STEP", 5000, "maximum steps for agent-env. interaction")
flags.DEFINE_integer("TRAINING_EPISODE", 3000, "maximum episodes for agent-env. interaction")
flags.DEFINE_integer("UPDATE_PERIOD", 10000000, "update_period")

def random_choice(prob):
  # Randomly choose item for given probability.
  # 
  # Args:
  # - prob: probability (2-D list)
  c = np.cumsum(prob, axis = 1)
  u = np.random.rand(len(c), 1)
  
  return (u < c).argmax(axis = 1)

class GymAgent(object):
  def __init__(self, env):
    # Initialize the agent in OpenAI Gym environment.
    #
    # Args:
    # - env: environment
    self.env = env
    self.FLAGS = tf.app.flags.FLAGS

    """ Observation size """
    self.obs_size = env.observation_space.shape[0]

    """ Action size """
    try: # Box
      self.action_size = env.action_space.shape
    except AttributeError: # Discrete
      self.action_size = env.action_space.n

    """ TensorFlow graph construction """
    self.build_model()
    self.build_loss()
    self.build_optimizer()

    """ Open Tensorflow session and initialize variables """
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

  def build_model(self):
    self.obs = tf.placeholder(tf.float32, [None, self.obs_size])

    def _model(net):
      with slim.arg_scope([fc], weights_initializer = tf.random_normal_initializer(0.0001)):
        net = fc(net, 16, activation_fn = tf.nn.sigmoid, scope = 'fc0')
        net = fc(net, self.action_size, activation_fn = tf.nn.softmax, scope = 'fc1')
      return net

    self.policy = _model(self.obs)

  def build_loss(self):
    self.actions = tf.placeholder(tf.int32, [None])
    self.returns = tf.placeholder(tf.float32, [None])

    def _loss():
      log_policy = tf.clip_by_value(self.policy, 1e-20, 1.0)
      actions = tf.one_hot(self.actions, self.action_size)
      loss = -tf.reduce_sum(tf.reduce_sum(log_policy * actions, axis = 1) * self.returns)
      return loss

    self.loss = _loss()

  def build_optimizer(self):
    self.optimizer = tf.train.GradientDescentOptimizer(self.FLAGS.LEARNING_RATE).minimize(self.loss)

  def act(self, obs):
    # Choose action based on observation.
    #
    # Args:
    # - obs: observation
    # - Note that single observation is assumed here.
    policy = self.sess.run(self.policy, feed_dict = {self.obs: [obs]})
    action = random_choice(policy)

    return int(action[0])

  def learn(self, mode = 0):
    # Args:
    # - mode: training mode
    FLAGS = self.FLAGS
    training_step = FLAGS.TRAINING_STEP

    n_step_episode = 0
    n_episode = 0
    n_total_step = 0
    expr_buf = [[], [], [], [], []]

    def _insert_expr(expr_buf, *expr):
      for i in range(len(expr)):
        expr_buf[i].append(expr[i])
      return expr_buf

    def _episode_ends(obs, n_total_step, n_episode, n_step_episode):
      print "training\t|step: {0}\t|episode: {1}\t|length: {2}".format(
          n_total_step, n_episode, n_step_episode)
      obs = self.env.reset()
      return obs, n_total_step, n_episode + 1, 0

    env = self.env
    obs = env.reset()

    if mode == 1:
      for episode in range(FLAGS.TRAINING_EPISODE):
        while True:
          action = self.act(obs)
          obs_n, reward, done, _ = env.step(action)
          expr_buf = _insert_expr(expr_buf, obs, action, reward, obs_n, done)
          if len(expr_buf[0]) >= FLAGS.UPDATE_PERIOD or done\
              or n_step_episode >= env.spec.timestep_limit:
            self.update(expr_buf)
            expr_buf = [[], [], [], [], []]
          obs = obs_n

          n_step_episode += 1
          n_total_step += 1

          if done or n_step_episode >= env.spec.timestep_limit:
            break

        obs, n_total_step, n_episode, n_step_episode =\
            _episode_ends(obs, n_total_step, n_episode, n_step_episode)

    else:
      raise NotImplementedError()

  def update(self, expr_buf):
    # REINFORCE without baseline
    FLAGS = self.FLAGS

    rewards = np.array(expr_buf[2])
    rewards = rewards * (FLAGS.GAMMA ** np.arange(len(rewards)))
    returns = rewards[::-1].cumsum()[::-1]
#    try:
#      self.baseline.append(returns[0])
#    except AttributeError:
#      self.baseline = []
#      self.baseline.append(returns[0])
#
#    self.baseline = self.baseline[-100:]
#    baseline = sum(self.baseline) / len(self.baseline)
#
    for i in range(len(expr_buf[0])):
      feed_dict = {self.obs: [expr_buf[0][i]],
                   self.actions: [expr_buf[1][i]],
                   self.returns: [returns[i]]}
      self.sess.run(self.optimizer, feed_dict = feed_dict)

def main():
  """ Environment setting """
  env = gym.make("CartPole-v0")
  input_layer_size = env.observation_space.shape[0]
  hidden_layer_size = 16

  """ Agent setting """
  agent = GymAgent(env)

  """ Learning """
  # mode: 0
  # - Use FLAGS.TRAINING_STEP.
  # mode: 1
  # - Use FLAGS.TRAINING_EPISODE.
  agent.learn(mode = 1)

  agent.sess.close()

if __name__ == "__main__":
  main()
