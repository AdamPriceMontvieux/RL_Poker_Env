from agents.base_agent import BaseAgent

class HeuristicAgent(BaseAgent):

    def __init__(self, ID):
        super.__init__(ID)
        

    def get_action(self, observation):
        """gets an action from the agent that should be performed based on the agent's
         internal state and provided observation and action space"""
        raise NotImplementedError
    
    def process_observation(self, observation):
        """Process an observation vector"""
        raise NotImplementedError



