import numpy as np


class PBRS:
    def __init__(self, n_states):
        self.n_states = n_states
        self.discount = 0.9
        self.state_potentials = np.zeros(n_states)
        self._initial_potentials = np.zeros(n_states)  # Used for resetting potentials to original seed

    def potential_function(self, current_state, prev_state):
        """
        Calculates change in potential over a state transition
        """
        phi_s1 = self.state_potentials[prev_state]
        phi_s2 = self.discount*self.state_potentials[current_state]
        delta_potential = phi_s2 - phi_s1

        return delta_potential

    def reset_potentials(self):
        """
        Reset the state potential estimates
        """
        self.state_potentials = self._initial_potentials.copy()

    def random_init(self):
        """
        Create a random initialization for state potentials
        """
        self._initial_potentials = np.random.normal(0, 1, self.n_states)
        self.state_potentials = self._initial_potentials.copy()

    def set_potentials(self, gw, agent_id, n_steps, ptype):
        """
        Set the potentials of states based on target locations
        """
        if ptype == "exploration":
            self.exploration_potential(gw, n_steps)
        elif ptype == "target_prox":
            self.target_proximity_potential(gw, n_steps)
        elif ptype == "target_agent":
            self.target_agent_distance(gw, agent_id)
        elif ptype == "custom":
            self.custom_potential(gw, agent_id)

    def exploration_potential(self, gw, nsteps):
        """
        Potential function that incentivizes agents travelling away from the center and towards targets
        """
        center_x = int(gw.width/2)
        center_y = int(gw.height/2)

        for x in range(gw.width):
            for y in range(gw.height):
                state = x + gw.height * y
                x_dist = abs(x - center_x)
                y_dist = abs(y - center_y)
                self._initial_potentials[state] = (x_dist + y_dist)/nsteps

        self.state_potentials = self._initial_potentials.copy()

    def custom_potential(self, gw, agent_id):
        """
        Use customized potential function (assign 1 agent to a target)
        """
        poi_state_potentials = [[] for i in range(self.n_states)]
        agent_loc = gw.agents[f'A{agent_id}'].initial_position
        for t_id, t_loc in enumerate(gw.targets):
            for x in range(gw.width):
                for y in range(gw.height):
                    state = x + gw.height * y
                    target_state_dist = abs(x - t_loc[0]) + abs(y - t_loc[1])
                    state_agent_dist = abs(agent_loc[0] - x) + abs(agent_loc[1] - y)
                    target_agent_dist = abs(agent_loc[0] - t_loc[0]) + abs(agent_loc[1] - t_loc[1])

                    if (target_state_dist + state_agent_dist) == target_agent_dist and t_id == agent_id:
                        potential = gw.target_values[t_id]*(1 - (target_state_dist / target_agent_dist))
                    else:
                        potential = 0.0
                    poi_state_potentials[state].append(potential)

        for state in range(self.n_states):
            self._initial_potentials[state] = max(poi_state_potentials[state])
        self.state_potentials = self._initial_potentials.copy()

    def target_proximity_potential(self, gw, n_steps):
        """
        Potential function that evaluates state potential based on proximity to nearby targets
        """
        poi_state_potentials = [[] for i in range(self.n_states)]
        for t_id, t_loc in enumerate(gw.targets):
            for x in range(gw.width):
                for y in range(gw.height):
                    state = x + gw.height * y
                    x_dist = abs(x - t_loc[0])
                    y_dist = abs(y - t_loc[1])
                    poi_state_distance = x_dist + y_dist
                    potential = (1 - (poi_state_distance/n_steps))
                    poi_state_potentials[state].append(potential)

        for state in range(self.n_states):
            self._initial_potentials[state] = np.mean(poi_state_potentials[state])
        self.state_potentials = self._initial_potentials.copy()

    def target_agent_distance(self, gw, agent_id):
        """
        Potential function that incentivizes agent travelling directly towards targets
        """
        poi_state_potentials = [[] for i in range(self.n_states)]
        agent_loc = gw.agents[f'A{agent_id}'].initial_position
        for t_id, t_loc in enumerate(gw.targets):
            for x in range(gw.width):
                for y in range(gw.height):
                    state = x + gw.height * y
                    target_state_dist = abs(x - t_loc[0]) + abs(y - t_loc[1])
                    state_agent_dist = abs(agent_loc[0] - x) + abs(agent_loc[1] - y)
                    target_agent_dist = abs(agent_loc[0] - t_loc[0]) + abs(agent_loc[1] - t_loc[1])

                    if (target_state_dist + state_agent_dist) == target_agent_dist:
                        potential = 1 - (target_state_dist / (target_state_dist + state_agent_dist))
                    else:
                        potential = 0.0
                    poi_state_potentials[state].append(potential)

        for state in range(self.n_states):
            self._initial_potentials[state] = max(poi_state_potentials[state])
        self.state_potentials = self._initial_potentials.copy()
