class BaseAgent:
    
    def train(self):
        """allows an agent to learn a policy"""
        raise NotImplementedError

    def get_action(self, observation, action_space):
        """gets an action from the agent that should be performed based on the agent's
         internal state and provided observation and action space"""
        raise NotImplementedError
    
    def process_observation(self, observation):
        """Process an observation vector"""
        raise NotImplementedError


