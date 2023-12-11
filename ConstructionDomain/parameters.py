parameters = {}

# Test Parameters
parameters["starting_srun"] = 0  # Which stat run should testing start on (used for parallel testing)
parameters["stat_runs"] = 1  # Total number of runs to perform
parameters["generations"] = 1000  # Number of generations for CCEA in each stat run
parameters["algorithm"] = "Global"  # Global, Difference, DPP (D++), CFL
parameters["sample_rate"] = 20  # Spacing for collecting performance data during training (every X generations)
parameters["n_configurations"] = 1  # The number of environmental configurations used for training
parameters["c_type"] = "Custom"  # Auto or Custom

# Domain parameters
parameters["x_dim"] = 30.0  # X-Dimension of the environment
parameters["y_dim"] = 30.0  # Y-Dimension of the environment
parameters["n_rovers"] = 2  # Number of standard rovers
parameters["n_excavators"] = 2  # Number of excavators
parameters["n_agents"] = parameters["n_rovers"] + parameters["n_excavators"]  # Number of total agents
parameters["n_dig_sites"] = 3   # Number of dig sites in the environment
parameters["steps"] = 20  # Number of time steps rovers take each episode
parameters["world_setup"] = "All"  # Agent_Only, All
parameters["ds_config_type"] = "Random"  # Random, Two_DS_LR, Twp_DS_TB, Four_Corners, Circle
parameters["agent_config_type"] = "Concentrated"  # Random, Concentrated, Fixed

# Dig-site Parameters
parameters["observation_radius"] = 3.0  # Maximum range at which rovers can mark a Dig Site
parameters["excavation_radius"] = 2.0  # Maximum range at which an excavator can excavate a dig site

# Agent Parameters
parameters["sensor_model"] = "summed"  # Should either be "density" or "summed"
parameters["angle_res"] = 360 / 4  # Resolution of sensors (determines number of sectors)
parameters["sensor_range"] = None  # Standard rover sensor range (None is infinite)

# Rover Parameters
parameters["rdmax"] = 1.0  # Maximum distance a rover can move in a single time step
parameters["rv_sensor_range"] = None  # Standard rover sensor range (None is infinite)

# Excavator Parameters
parameters["edmax"] = 0.5 * parameters["rdmax"]  # Maximum distance an excavator can move in a single time step
parameters["ex_sensor_range"] = None  # Sensor range of excavators (None is infinite)

# Neural network parameters for rover motor control
parameters["n_inp"] = int(2 * (360/parameters["angle_res"]))
parameters["n_hid"] = 12
parameters["n_out"] = 2

# CCEA parameters
parameters["pop_size"] = 40
parameters["mutation_chance"] = 0.1  # Probability that a mutation will occur
parameters["mutation_rate"] = 0.2  # How much a weight is allowed to change
parameters["epsilon"] = 0.1  # For e-greedy selection in CCEA
parameters["n_elites"] = 1  # How many elites to carry over during elite selection

# Post Training Test Parameters
parameters["c_list_size"] = 10000
parameters["vis_running"] = True  # True keeps visualizer from closing until you 'X' out of window
