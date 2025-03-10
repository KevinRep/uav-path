import numpy as np

class RandomAgent:
    def __init__(self, env):
        self.env = env
    
    def select_action(self, state):
        return self.env.action_space.sample()