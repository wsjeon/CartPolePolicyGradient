def get_agent(agent_type):
  if agent_type == "reinforce":
    from .reinforce import REINFORCE; agent = REINFORCE
  else:
    raise NotImplementedError()

  return agent
