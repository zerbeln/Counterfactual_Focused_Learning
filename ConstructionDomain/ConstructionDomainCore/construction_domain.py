import numpy as np
import csv
import copy
from parameters import parameters as p
from ConstructionDomainCore.rover import Rover
from ConstructionDomainCore.excavator import Excavator
from ConstructionDomainCore.digsite import DigSite


class ConstructionDomain:
    def __init__(self):
        # World attributes
        self.world_x = p["x_dim"]
        self.world_y = p["y_dim"]
        self.n_dig_sites = p["n_dig_sites"]
        self.n_agents = p["n_agents"]
        self.n_rovers = p["n_rovers"]
        self.n_excavators = p["n_excavators"]
        self.obs_radius = p["observation_radius"]  # Maximum distance rovers can successfully mark a Dig Site
        self.ex_radius = p["excavation_radius"]  # Maximum distance excavators can successfully excavate a Dig Site
        self.agent_ds_distances = [[] for i in range(self.n_dig_sites)]  # Tracks rover distances to dig sites at each time step
        self.excavator_ds_distance = [[] for i in range(self.n_dig_sites)]  # Tracks excavator distances to dig sites at each time step

        # Agent Instances and Configurations
        self.agents = {}  # Dictionary containing instances of agent objects
        self.agent_configurations = [[] for _ in range(p["n_agents"])]

        # Dig Site Instances
        self.dig_sites = {}  # Dictionary containing instances of dig site objects
        self.dig_site_configurations = [[] for _ in range(p["n_dig_sites"])]

    def reset_world(self, cf_id):
        """
        Reset world to initial conditions.
        """
        self.agent_ds_distances = [[] for i in range(self.n_dig_sites)]
        self.excavator_ds_distances = [[] for i in range(self.n_dig_sites)]
        for ag in self.agents:
            self.agents[ag].reset_agent(self.agent_configurations[self.agents[ag].agent_id][cf_id])
        for ds in self.dig_sites:
            self.dig_sites[ds].reset_ds(self.dig_site_configurations[self.dig_sites[ds].ds_id][cf_id])

    def load_world(self):
        """
        Load information from a saved csv file.
        """
        # Initialize Dig Site positions and values
        self.load_ds_configuration()

        # Initialize Agent Positions
        self.load_agent_configuration()

    def calc_global(self):
        """
        Calculate the global reward for the current state as the reward given by each Dig Site.
        :return: Array capturing reward given from each Dig Site at current time step (sum is the global reward)
        """
        global_reward = np.zeros(self.n_dig_sites)

        for ds in self.dig_sites:
            # Update global reward if Dig Site is marked and excavated
            if self.dig_sites[ds].marked and self.dig_sites[ds].excavated:
                global_reward[self.dig_sites[ds].ds_id] = self.dig_sites[ds].value

        return global_reward

    def load_ds_configuration(self):
        """
        Load Dig Site configuration from a CSV file
        """

        for cf_id in range(p["n_configurations"]):
            csv_input = []
            with open(f'./World_Config/DigSite_Config{cf_id}.csv') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')

                for row in csv_reader:
                    csv_input.append(row)

            for ds_id in range(self.n_dig_sites):
                ds_x = float(csv_input[ds_id][0])
                ds_y = float(csv_input[ds_id][1])
                ds_val = float(csv_input[ds_id][2])
                ds_coupling = float(csv_input[ds_id][3])

                if cf_id == 0:
                    self.dig_sites[f'P{ds_id}'] = DigSite(ds_x, ds_y, ds_val, ds_coupling, ds_id)

                self.dig_site_configurations[ds_id].append((ds_x, ds_y, ds_val, ds_coupling))

    def load_agent_configuration(self):
        """
        Load agent configuration from a saved csv file
        """

        for cf_id in range(p["n_configurations"]):
            csv_input = []
            with open(f'./World_Config/Agent_Config{cf_id}.csv') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')

                for row in csv_reader:
                    csv_input.append(row)

            for agent_id in range(self.n_agents):
                agent_x = float(csv_input[agent_id][0])
                agent_y = float(csv_input[agent_id][1])
                agent_theta = float(csv_input[agent_id][2])
                self.agent_configurations[agent_id].append((agent_x, agent_y, agent_theta))

                if agent_id < self.n_agents - self.n_excavators:
                    if cf_id == 0:
                        self.agents[f'R{agent_id}'] = Rover(agent_id, agent_x, agent_y, agent_theta)
                else:
                    if cf_id == 0:
                        self.agents[f'E{agent_id}'] = Excavator(agent_id, agent_x, agent_y, agent_theta)

    def step(self, agent_actions):
        """
        Environment takes in agent actions and returns next state and the global reward
        :param agent_actions: array containing actions for each agent taken from agent neural networks
        :return global_reward: numpy array containing the reward given by each Dig Site for the given state
        """

        # Agents take action from neural network
        for ag in self.agents:
            dx = 2 * self.agents[ag].dmax * (agent_actions[self.agents[ag].agent_id][0] - 0.5)
            dy = 2 * self.agents[ag].dmax * (agent_actions[self.agents[ag].agent_id][1] - 0.5)

            # Calculate new agent X Position
            x = dx + self.agents[ag].loc[0]

            # Agents cannot move beyond boundaries of the world
            if x < 0:
                x = 0
            elif x > self.world_x - 1:
                x = self.world_x - 1

            # Calculate new agent Y Position
            y = dy + self.agents[ag].loc[1]

            # Agents cannot move beyond boundaries of the world
            if y < 0:
                y = 0
            elif y > self.world_y - 1:
                y = self.world_y - 1

            # Update agent position
            self.agents[ag].loc[0] = x
            self.agents[ag].loc[1] = y

        # Agents observe the new world state
        for ag in self.agents:
            self.agents[ag].scan_environment(self.agents, self.dig_sites)

        # REWARD CALCULATION ------------------------------------------------------------------------------------------
        # Record agent distances from dig sites for given time step (for reward calculations)
        for ds in self.dig_sites:
            self.dig_sites[ds].update_dig_sites(self.agents)
            self.agent_ds_distances[self.dig_sites[ds].ds_id].append(self.dig_sites[ds].agent_distances)

        global_reward = self.calc_global()

        return global_reward
