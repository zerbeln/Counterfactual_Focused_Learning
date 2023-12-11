from ConstructionDomainCore.agent import Agent
from parameters import parameters as p


class Excavator(Agent):
    def __init__(self, exc_id, exc_x, exc_y, exc_theta):
        super().__init__(exc_id, exc_x, exc_y, exc_theta)
        self.type = "Excavator"  # Excavator type agent
        self.dmax = p["edmax"]  # Excavator moves half as fast as a rover
        self.sensor_range = p["ex_sensor_range"]

