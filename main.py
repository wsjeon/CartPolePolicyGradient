import tensorflow as tf
import gym
from agents import get_agent

flags = tf.app.flags

flags.DEFINE_string("ENVIRONMENT", "CartPole-v0", "environment")
flags.DEFINE_string("AGENT", "reinforce", "agent")
flags.DEFINE_float("GAMMA", 0.98, "discount factor")
flags.DEFINE_float("LEARNING_RATE", 0.01, "learning rate")
flags.DEFINE_integer("TRAINING_STEP", 5000, "maximum steps for agent-env. interaction")
flags.DEFINE_integer("TRAINING_EPISODE", 2000, "maximum episodes for agent-env. interaction")

FLAGS = flags.FLAGS

def main():
  """ Environment setting """
  env = gym.make("CartPole-v0")

  """ Agent setting """
  from agents import get_agent; Agent = get_agent(FLAGS.AGENT)
  agent = Agent(env)

  """ Learning """
  # mode: 0
  # - Use FLAGS.TRAINING_STEP.
  # mode: 1
  # - Use FLAGS.TRAINING_EPISODE.
  agent.learn(mode = 1, monitor = True)
  agent.sess.close()

if __name__ == "__main__":
  main()
