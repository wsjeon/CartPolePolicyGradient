import tensorflow as tf
import os

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
    self.build_net()
    self.build_loss()
    self.build_optimizer()
    self.build_summary()

    """ Open Tensorflow session and initialize variables """
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

  def build_net(self):
    raise NotImplementedError()

  def build_loss(self):
    raise NotImplementedError()

  def build_optimizer(self):
    raise NotImplementedError()

  def build_summary(self):
    self.return_episode = tf.placeholder(tf.float32)
    tf.summary.scalar("score", self.return_episode)
    self.summary_op = tf.summary.merge_all()

    run_count = 0
    while True:
      tmp_dir = os.path.join(self.FLAGS.SUMMARY_DIR, "run{}".format(int(run_count)))
      if os.path.exists(tmp_dir):
        run_count += 1
      else:
        break
    self.summary_writer = tf.summary.FileWriter(tmp_dir)

  def update(self):
    raise NotImplementedError()

  def act(self):
    raise NotImplementedError()

  def learn(self):
    raise NotImplementedError()

