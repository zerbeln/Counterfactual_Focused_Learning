import numpy as np
import sys
from parameters import parameters as p
from ConstructionDomainCore.agent import Agent

class Rover(Agent):
    def __init__(self, rov_id, rov_x, rov_y, rov_theta):
        super().__init__(rov_id, rov_x, rov_y, rov_theta)

        # Rover Paramters
        self.type = "Rover"  # Standard rover type of agent
        self.dmax = p["rdmax"]  # Maximum distance a rover can move each time step
        self.sensor_range = p["rv_sensor_range"]  # Distance rovers can perceive environment



