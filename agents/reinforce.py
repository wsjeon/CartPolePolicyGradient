# REINFORCE without baseline
# Eq. (21) at:
# http://www.keck.ucsf.edu/~houde/sensorimotor_jc/possible_papers/JPeters08a.pdf
# Reference code at:
# https://gym.openai.com/evaluations/eval_PqFieTTRCCwtOx96e8KMw
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import fully_connected as fc
from gym_agent import GymAgent

class REINFORCE(GymAgent):
  def __init__(self, env):
    GymAgent.__init__(self, env)

  def build_net(self):
    self.obs = tf.placeholder(tf.float32, [None, self.obs_size])

    def _net(net, hidden_layer_size = 16):
      weights_init1 = tf.random_normal_initializer(stddev = 1 / np.sqrt(self.obs_size))
      weights_init2 = tf.random_normal_initializer(stddev = 1 / np.sqrt(hidden_layer_size))
      net = fc(net, hidden_layer_size, activation_fn = tf.nn.sigmoid, scope = 'fc0',
               weights_initializer = weights_init1)
      net = fc(net, self.action_size, activation_fn = tf.nn.softmax, scope = 'fc1',
               weights_initializer = weights_init2)
      return net

    self.policy = _net(self.obs)

  def build_loss(self):
    self.actions = tf.placeholder(tf.int32, [None])
    self.advantage = tf.placeholder(tf.float32)

    def _loss():
      log_policy = tf.log(self.policy)
      actions = tf.one_hot(self.actions, self.action_size)
      loss = - tf.reduce_sum(log_policy * actions) * self.advantage
      return loss

    self.loss = _loss()

  def build_optimizer(self):
    opt = tf.train.GradientDescentOptimizer(self.FLAGS.LEARNING_RATE)
    self.optimizer = opt.minimize(self.loss)

  def update(self, expr_buf):
    FLAGS = self.FLAGS

    rewards = np.array(expr_buf[2])
    discounted_rewards = rewards * np.power(FLAGS.GAMMA, np.arange(len(rewards)))
    return_episode = discounted_rewards.sum()

    try:
      self.returns.append(return_episode)
    except AttributeError:
      self.returns = []
      self.returns.append(return_episode)
    self.returns = self.returns[-100:]
    baseline = sum(self.returns) / len(self.returns)

    feed_dict = {self.obs: expr_buf[0],
                 self.actions: expr_buf[1],
                 self.advantage: return_episode - baseline}

    self.sess.run(self.optimizer, feed_dict = feed_dict)

  def act(self, obs):
    # Choose action based on observation.
    #
    # Args:
    # - obs: observation
    # - Note that single observation is assumed here.
    policy = self.sess.run(self.policy, feed_dict = {self.obs: [obs]})
    return np.random.choice(self.action_size, p = policy[0])

  def learn(self, mode = 1, monitor = False, monitor_dir = "./monitor"):
    # Args:
    # - mode: training mode
    FLAGS = self.FLAGS

    def _insert_expr(expr_buf, *expr):
      for i in range(len(expr)):
        expr_buf[i].append(expr[i])
      return expr_buf

    def _episode_ends(obs, n_total_step, n_episode, n_step_episode, return_episode):
      print "training\t|step: {0}\t|episode: {1}\t|length: {2}\t|return: {3}".format(
          n_total_step, n_episode, n_step_episode, return_episode)
      summary_str = self.sess.run(self.summary_op, feed_dict = {self.return_episode: return_episode})
      self.summary_writer.add_summary(summary_str, n_episode)
      self.summary_writer.flush()

      obs = self.env.reset()
      return obs, n_total_step, n_episode + 1, 0, 0.0

    env = self.env

    if monitor:
      env.monitor.start(monitor_dir, force = True)

    obs = env.reset()

    n_total_step = 0
    n_episode = 0
    n_step_episode = 0
    expr_buf = [[], [], [], [], []]

    if mode == 1:
      for episode in range(FLAGS.TRAINING_EPISODE):
        return_episode = 0.0
        while True:
          action = self.act(obs)
          obs_n, reward, done, _ = env.step(action)
          expr_buf = _insert_expr(expr_buf, obs, action, reward, obs_n, done)
          if done or n_step_episode >= env.spec.timestep_limit:
            self.update(expr_buf)
            expr_buf = [[], [], [], [], []]
          obs = obs_n

          n_total_step += 1
          n_step_episode += 1
          return_episode += reward

          if done or n_step_episode >= env.spec.timestep_limit:
            break

        obs, n_total_step, n_episode, n_step_episode, return_episode =\
            _episode_ends(obs, n_total_step, n_episode, n_step_episode, return_episode)

    else:
      raise NotImplementedError()

    if monitor:
      env.monitor.close()
