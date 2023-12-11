from global_functions import get_linear_dist
from parameters import parameters as p
import numpy as np


class DigSite:
    def __init__(self, dsx, dsy, ds_val, coupling, ds_id):
        self.ds_id = ds_id  # Dig Site Identifier
        self.loc = [dsx, dsy]  # Location of the Dig Site
        self.value = ds_val  # Dig Site Value
        self.marking_dist = p["observation_radius"]
        self.excavation_dist = p["excavation_radius"]

        self.agent_distances = np.zeros(p["n_agents"])  # Distance between dig site and each agent
        self.marked = False  # Boolean that indicates if a dig site has been successfully marked
        self.excavated = False  # Boolean that indicates if a dig site has been successfully excavated
        self.quadrant = None  # Tracks which quadrant (or sector) of the environment a Dig Site exists in

    def reset_ds(self, ds_config):
        """
        Clears the observer distances array and sets dig site marked/excavated booleans back to False
        """
        self.loc[0] = ds_config[0]
        self.loc[1] = ds_config[1]
        self.value = ds_config[2]
        self.agent_distances = np.zeros(p["n_agents"])  # Distance between dig site and each agent
        self.marked = False
        self.excavated = False

    def update_dig_sites(self, agents):
        """
        Determine if digsites are marked or excavated
        :param agents: Dictionary containing Rover and Excavator class instances
        """
        for ag in agents:
            dist = get_linear_dist(agents[ag].loc[0], agents[ag].loc[1], self.loc[0], self.loc[1])
            self.agent_distances[agents[ag].agent_id] = dist

            if agents[ag].type == "Rover" and dist < self.marking_dist:
                self.marked = True

            # Dig sites must be marked before excavation
            if self.marked and agents[ag].type == "Excavator" and dist < self.excavation_dist:
                self.excavated = True
