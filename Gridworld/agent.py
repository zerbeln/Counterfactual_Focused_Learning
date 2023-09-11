import random
import numpy as np


class Agent:
    def __init__(self, x, y):
        self.loc = [x, y]
        self.initial_position = (x, y)
        self.actions = [0, 1, 2, 3, 4]  # Up, Down, Left, Right, Stationary

    def reset_agent(self):
        """
        Reset the agent to its initial position
        """
        self.loc[0] = self.initial_position[0]
        self.loc[1] = self.initial_position[1]


class QLearner(Agent):
    def __init__(self, n_states, x, y):
        super().__init__(x, y)
        self.n_states = n_states
        self.n_actions = len(self.actions)
        self.discount = 0.9  # Discount factor (gamma)
        self.alpha = 0.1  # Learning rate
        self.epsilon = 0.15  # e-greedy
        self.q_table = np.zeros((n_states, len(self.actions)))
        self.current_state = None
        self.prev_state = None
        self.action = None

    def set_current_state(self, state):
        """
        Set agents current state after resetting for new epoch
        """
        self.prev_state = None
        self.current_state = state

    def update_state(self, new_state):
        """
        Updates the agents current and previous states after taking an action
        """
        self.prev_state = self.current_state
        self.current_state = new_state

    def update_q_val(self, reward):
        """
        Update q-values after a state transition
        """
        q_val = self.q_table[self.prev_state, self.action]
        max_q = np.max(self.q_table[self.current_state])

        new_q = ((1-self.alpha)*q_val) + self.alpha*(reward + (self.discount*max_q) - q_val)
        self.q_table[self.prev_state, self.action] = new_q

    def get_egreedy_action(self, state):
        """
        Choose an action with e-greedy selection
        """
        rand_val = random.uniform(0, 1)
        if rand_val > self.epsilon:
            return np.argmax(self.q_table[state])
        else:
            return random.randint(0, self.n_actions-1)

    def get_greedy_action(self, state):
        """
        Only choose the action with the highest value estimate
        """
        return np.argmax(self.q_table[state])

    def reset_learner(self):
        """
        Clear data in the q-table
        """
        self.q_table = np.zeros((self.n_states, self.n_actions))
