import numpy as np
import sys
from parameters import parameters as p
from global_functions import get_squared_dist, get_angle


class Agent:
    def __init__(self, ag_id, ag_x, ag_y, ag_theta):
        # Agent Parameters -----------------------------------------------------------------------------------
        self.agent_id = ag_id  # Agent identifier
        self.loc = [ag_x, ag_y, ag_theta]  # Agent location

        # Agent Sensor Characteristics -----------------------------------------------------------------------
        self.sensor_res = p["angle_res"]  # Angular resolution of the sensors
        self.sensor_type = p["sensor_model"]  # Type of sensors agent is equipped with
        self.sensor_range = p["sensor_range"]  # Distance agents can perceive environment objects
        self.n_inputs = p["n_inp"]  # Number of inputs for rover's neural network

        # Agent Data -----------------------------------------------------------------------------------------
        self.observations = np.zeros(p["n_inp"], dtype=np.longdouble)  # Number of sensor inputs for Neural Network
        self.agent_actions = np.zeros(p["n_out"], dtype=np.longdouble)  # Motor actions from neural network outputs

    def reset_agent(self, agent_config):
        """
        Resets the rover to its initial position in the world and clears observation array of state information
        """
        self.loc[0] = agent_config[0]
        self.loc[1] = agent_config[1]
        self.loc[2] = agent_config[2]
        self.observations = np.zeros(self.n_inputs, dtype=np.longdouble)

    def scan_environment(self, agents, dig_sites):
        """
        Constructs the state information that gets passed to the rover's neuro-controller
        :param agents: Dictionary containing rover and excavator class instances
        :param dig_sites: Dictionary containing Dig Site class instances
        """
        n_brackets = int(360.0 / self.sensor_res)
        ds_state = self.ds_scan(dig_sites, n_brackets)
        agent_state = self.agent_scan(agents, n_brackets)

        for i in range(n_brackets):
            self.observations[i] = ds_state[i]
            self.observations[n_brackets + i] = agent_state[i]

    def ds_scan(self, dig_sites, n_brackets):
        """
        Rover observes Dig Sites in the environment using sensors
        :param dig_sites: Dictionary containing Dig Site class instances
        :param n_brackets: integer value for the number of brackets/sectors rover sensors scan (resolution)
        :return ds_state: numpy array containing state information for Dig Site observations
        """
        ds_state = np.zeros(n_brackets)
        temp_ds_dist_list = [[] for _ in range(n_brackets)]

        # Log Dig Site distances into brackets
        for ds in dig_sites:
            angle = get_angle(dig_sites[ds].loc[0], dig_sites[ds].loc[1], (p["x_dim"]/2), (p["y_dim"]/2))
            dist = get_squared_dist(dig_sites[ds].loc[0], dig_sites[ds].loc[1], self.loc[0], self.loc[1])

            # If dig site is within sensor range -> add to list
            if self.sensor_range is not None and dist < self.sensor_range:
                bracket = int(angle / self.sensor_res)
                if bracket > n_brackets-1:
                    bracket -= n_brackets
                temp_ds_dist_list[bracket].append(dig_sites[ds].value/dist)

        # Encode Dig Site information into the state vector
        for bracket in range(n_brackets):
            if len(temp_ds_dist_list[bracket]) > 0:
                if self.sensor_type == 'density':
                    ds_state[bracket] = sum(temp_ds_dist_list[bracket])/len(temp_ds_dist_list[bracket])  # Density Sensor
                elif self.sensor_type == 'summed':
                    ds_state[bracket] = sum(temp_ds_dist_list[bracket])  # Summed Distance Sensor
                else:
                    sys.exit('Incorrect sensor model')
            else:
                ds_state[bracket] = -1.0

        return ds_state

    def agent_scan(self, agents, n_brackets):
        """
        Agent observes other agents in the environment using sensors
        :param agents: Dictionary containing rover and excavator class instances
        :param n_brackets: integer value for the number of brackets/sectors rover sensors scan (resolution)
        :return agent_state: numpy array containing state information for agent observations
        """
        agent_state = np.zeros(n_brackets)
        temp_rover_dist_list = [[] for _ in range(n_brackets)]

        # Log Rover distances into brackets
        for ag in agents:
            if self.agent_id != agents[ag].agent_id:  # Ignore self
                ag_x = agents[ag].loc[0]
                ag_y = agents[ag].loc[1]

                angle = get_angle(ag_x, ag_y, p["x_dim"]/2, p["y_dim"]/2)
                dist = get_squared_dist(ag_x, ag_y, self.loc[0], self.loc[1])

                # If rover is within sensor range -> add to list
                if self.sensor_range is not None and dist < self.sensor_range:
                    bracket = int(angle / self.sensor_res)
                    if bracket > n_brackets-1:
                        bracket -= n_brackets
                    temp_rover_dist_list[bracket].append(1/dist)

        # Encode Rover information into the state vector
        for bracket in range(n_brackets):
            if len(temp_rover_dist_list[bracket]) > 0:
                if self.sensor_type == 'density':
                    agent_state[bracket] = sum(temp_rover_dist_list[bracket]) / len(temp_rover_dist_list[bracket])  # Density Sensor
                elif self.sensor_type == 'summed':
                    agent_state[bracket] = sum(temp_rover_dist_list[bracket])  # Summed Distance Sensor
                else:
                    sys.exit('Incorrect sensor model')
            else:
                agent_state[bracket] = -1.0

        return agent_state
