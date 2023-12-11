from ConstructionDomainCore.construction_domain import ConstructionDomain
from Visualizer.visualizer import run_visualizer
import numpy as np
from ConstructionDomain.parameters import parameters as p
from ConstructionDomain.global_functions import create_pickle_file
import random
import math
import csv
import os
from ConstructionDomain.global_functions import get_linear_dist


def save_ds_configuration(dig_site_info, config_id):
    """
    Saves Dig Sites configuration to a csv file in a folder called World_Config
    """
    dir_name = './World_Config'  # Intended directory for output files

    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    pfile_name = os.path.join(dir_name, f'DigSite_Config{config_id}.csv')

    with open(pfile_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for ds_id in range(p["n_dig_sites"]):
            writer.writerow(dig_site_info[ds_id, :])

    csvfile.close()


def save_agent_configuration(initial_agent_positions, config_id):
    """
    Saves Agent configuration to a csv file in a folder called World_Config
    """
    dir_name = './World_Config'  # Intended directory for output files

    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    pfile_name = os.path.join(dir_name, f'Agent_Config{config_id}.csv')

    row = np.zeros(3)
    with open(pfile_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for agent_id in range(p["n_agents"]):
            row[0] = initial_agent_positions[agent_id, 0]
            row[1] = initial_agent_positions[agent_id, 1]
            row[2] = initial_agent_positions[agent_id, 2]
            writer.writerow(row[:])

    csvfile.close()


# ROVER POSITION FUNCTIONS ---------------------------------------------------------------------------------------
def agent_pos_random(dig_site_info):  # Randomly set rovers on map
    """
    Agents given random starting positions and orientations. Code ensures rovers do not start out too close to Dig Site.
    """
    initial_agent_positions = np.zeros((p["n_agents"], 3))

    for agent_id in range(p["n_agents"]):
        agent_x = random.uniform(0.0, p["x_dim"]-1.0)
        agent_y = random.uniform(0.0, p["y_dim"]-1.0)
        agent_theta = random.uniform(0.0, 360.0)
        buffer = 3  # Smallest distance to the outer Dig Site observation area a rover can spawn

        # Make sure agent does not start within observation range of a Dig Site
        agent_too_close = True
        while agent_too_close:
            count = 0
            for ds_id in range(p["n_dig_sites"]):
                dist = get_linear_dist(dig_site_info[ds_id, 0], dig_site_info[ds_id, 1], agent_x, agent_y)
                if dist < (p["observation_radius"] + buffer):
                    count += 1

            if count == 0:
                rover_too_close = False
            else:
                rover_x = random.uniform(0.0, p["x_dim"] - 1.0)
                rover_y = random.uniform(0.0, p["y_dim"] - 1.0)

        initial_agent_positions[agent_id, 0] = agent_x
        initial_agent_positions[agent_id, 1] = agent_y
        initial_agent_positions[agent_id, 2] = agent_theta

    return initial_agent_positions


def agent_pos_center_concentrated():
    """
    Agents given random starting positions within a radius of the center. Starting orientations are random.
    """
    radius = 8.0
    center_x = p["x_dim"]/2.0
    center_y = p["y_dim"]/2.0
    initial_agent_positions = np.zeros((p["n_agents"], 3))

    for agent_id in range(p["n_agents"]):
        x = random.uniform(0.0, p["x_dim"]-1.0)  # Agent X-Coordinate
        y = random.uniform(0.0, p["y_dim"]-1.0)  # Agent Y-Coordinate

        while x > (center_x + radius) or x < (center_x - radius):
            x = random.uniform(0.0, p["x_dim"]-1.0)  # Agent X-Coordinate
        while y > (center_y + radius) or y < (center_y - radius):
            y = random.uniform(0.0, p["y_dim"]-1.0)  # Agent Y-Coordinate

        initial_agent_positions[agent_id, 0] = x  # Agent X-Coordinate
        initial_agent_positions[agent_id, 1] = y  # Agent Y-Coordinate
        initial_agent_positions[agent_id, 2] = random.uniform(0.0, 360.0)  # Agent orientation

    return initial_agent_positions


def agent_pos_fixed_middle():  # Set rovers to fixed starting position
    """
    Agents start out extremely close to the center of the map (they may be stacked).
    """
    initial_agent_positions = np.zeros((p["n_agents"], 3))
    for agent_id in range(p["n_agents"]):
        initial_agent_positions[agent_id, 0] = 0.5*p["x_dim"] + random.uniform(-1.0, 1.0)
        initial_agent_positions[agent_id, 1] = 0.5*p["y_dim"] + random.uniform(-1.0, 1.0)
        initial_agent_positions[agent_id, 2] = random.uniform(0.0, 360.0)

    return initial_agent_positions


# DIG SITE POSITION FUNCTIONS ------------------------------------------------------------------------------------------
def ds_pos_random(coupling):  # Randomly set Dig Site on the map
    """
    Dig Site positions set randomly across the map (but not too close to other Dig Sites).
    """
    dig_site_info = np.zeros((p["n_dig_sites"], 4))  # [X, Y, Val, Coupling]

    for ds_id in range(p["n_dig_sites"]):
        x = random.uniform(0, p["x_dim"]-1.0)
        y = random.uniform(0, p["y_dim"]-1.0)

        # Make sure Dig Site don't start too close to one another
        ds_too_close = True
        while ds_too_close:
            count = 0
            for temp_id in range(p["n_dig_sites"]):
                if temp_id != ds_id:
                    x_dist = x - dig_site_info[temp_id, 0]
                    y_dist = y - dig_site_info[temp_id, 1]

                    dist = math.sqrt((x_dist**2) + (y_dist**2))
                    if dist < (p["observation_radius"] + 3.5):
                        count += 1

            if count == 0:
                ds_too_close = False
            else:
                x = random.uniform(0, p["x_dim"] - 1.0)
                y = random.uniform(0, p["y_dim"] - 1.0)

        dig_site_info[ds_id, 0] = x
        dig_site_info[ds_id, 1] = y
        dig_site_info[ds_id, 3] = coupling

    return dig_site_info


def ds_pos_circle(coupling):
    """
    Dig Sites positions are set in a circle around the center of the map at a specified radius.
    """
    dig_site_info = np.zeros((p["n_dig_sites"], 4))  # [X, Y, Val, Coupling]
    radius = 15.0
    interval = float(360/p["n_dig_sites"])

    x = p["x_dim"]/2.0
    y = p["y_dim"]/2.0
    theta = 0.0

    for ds_id in range(p["n_dig_sites"]):
        dig_site_info[ds_id, 0] = x + radius*math.cos(theta*math.pi/180)
        dig_site_info[ds_id, 1] = y + radius*math.sin(theta*math.pi/180)
        dig_site_info[ds_id, 3] = coupling
        theta += interval

    return dig_site_info


def ds_pos_two_ds_LR(coupling):
    """
    Sets two Dig Sites on the map, one on the left, one on the right in line with global X-axis.
    """
    assert(p["n_dig_sites"] == 2)
    dig_site_info = np.zeros((p["n_dig_sites"], 4))  # [X, Y, Val, Coupling]

    # Left Dig Site
    dig_site_info[0, 0] = 1.0
    dig_site_info[0, 1] = (p["y_dim"]/2.0) - 1
    dig_site_info[0, 3] = coupling

    # Right Dig Site
    dig_site_info[1, 0] = p["x_dim"] - 2.0
    dig_site_info[1, 1] = (p["y_dim"]/2.0) + 1
    dig_site_info[1, 3] = coupling

    return dig_site_info


def ds_pos_two_ds_TB(coupling):
    """
    Sets two Dig Sites on the map, one on the left, one on the right in line with global X-axis.
    """
    assert(p["n_dig_sites"] == 2)
    dig_site_info = np.zeros((p["n_dig_sites"], 4))  # [X, Y, Val, Coupling]

    # Top Dig Site
    dig_site_info[0, 0] = p["x_dim"]/2.0
    dig_site_info[0, 1] = 1
    dig_site_info[0, 3] = coupling

    # Bottom Dig Site
    dig_site_info[1, 0] = p["x_dim"]/2.0
    dig_site_info[1, 1] = p["y_dim"] - 1
    dig_site_info[1, 3] = coupling

    return dig_site_info


def ds_pos_four_corners(coupling):  # Statically set 4 Dig Sites (one in each corner)
    """
    Sets 4 Dig Sites on the map in a box formation around the center
    """
    assert(p["n_dig_sites"] == 4)  # There must only be 4 Dig Sites for this initialization
    dig_site_info = np.zeros((p["n_dig_sites"], 4))  # [X, Y, Val, Coupling]

    # Bottom left
    dig_site_info[0, 0] = 2.0
    dig_site_info[0, 1] = 2.0
    dig_site_info[0, 3] = coupling

    # Top left
    dig_site_info[1, 0] = 2.0
    dig_site_info[1, 1] = (p["y_dim"] - 2.0)
    dig_site_info[1, 3] = coupling

    # Bottom right
    dig_site_info[2, 0] = (p["x_dim"] - 2.0)
    dig_site_info[2, 1] = 2.0
    dig_site_info[2, 3] = coupling

    # Top right
    dig_site_info[3, 0] = (p["x_dim"] - 2.0)
    dig_site_info[3, 1] = (p["y_dim"] - 2.0)
    dig_site_info[3, 3] = coupling

    return dig_site_info


# DIG SITE VALUE FUNCTIONS -----------------------------------------------------------------------------------
def ds_vals_random(dig_site_info, v_low, v_high):
    """
    Dig Site values randomly assigned 1-10
    """
    for ds_id in range(p["n_dig_sites"]):
        dig_site_info[ds_id, 2] = float(random.randint(v_low, v_high))


def ds_vals_identical(dig_site_info, ds_val):
    """
    Dig Site values set to fixed, identical value
    """
    for ds_id in range(p["n_dig_sites"]):
        dig_site_info[ds_id, 2] = ds_val


def create_world_setup(coupling):
    """
    Create a new rover configuration file
    """
    for config_id in range(p["n_configurations"]):
        # Initialize Dig Site positions and values
        dig_site_info = np.zeros((p["n_dig_sites"], 4))  # [X, Y, Val, Coupling]

        if p["ds_config_type"] == "Random":
            dig_site_info = ds_pos_random(coupling)
            ds_vals_random(dig_site_info, 3, 10)
        elif p["ds_config_type"] == "Two_DS_LR":
            dig_site_info = ds_pos_two_ds_LR(coupling)
            ds_vals_identical(dig_site_info, 10.0)
        elif p["ds_config_type"] == "Two_DS_TB":
            dig_site_info = ds_pos_two_ds_TB(coupling)
            ds_vals_identical(dig_site_info, 10.0)
        elif p["ds_config_type"] == "Four_Corners":
            dig_site_info = ds_pos_four_corners(coupling)
            ds_vals_random(dig_site_info, 3.0, 10.0)
        elif p["ds_config_type"] == "Circle":
            dig_site_info = ds_pos_circle(coupling)
            ds_vals_random(dig_site_info, 3.0, 10.0)
        else:
            print("ERROR, WRONG DIG SITE CONFIG KEY")
        save_ds_configuration(dig_site_info, config_id)

        # Initialize Rover Positions
        initial_agent_positions = np.zeros((p["n_agents"], 3))  # [X, Y, Theta]

        if p["agent_config_type"] == "Random":
            initial_agent_positions = agent_pos_random(dig_site_info)
        elif p["agent_config_type"] == "Concentrated":
            initial_agent_positions = agent_pos_center_concentrated()
        elif p["agent_config_type"] == "Fixed":
            initial_agent_positions = agent_pos_fixed_middle()

        save_agent_configuration(initial_agent_positions, config_id)


def create_agent_setup_only():
    """
    Create new rover configurations while preserving the current Dig Site configurations
    """
    for cf_id in range(p["n_configurations"]):
        # Initialize Dig Site positions and values
        dig_site_info = np.zeros((p["n_dig_sites"], 4))  # [X, Y, Val, Coupling]

        config_input = []
        with open(f'./World_Config/DigSite_Config{cf_id}.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                config_input.append(row)

        for ds_id in range(p["n_dig_sites"]):
            ds_x = float(config_input[ds_id][0])
            ds_y = float(config_input[ds_id][1])
            ds_val = float(config_input[ds_id][2])
            ds_coupling = float(config_input[ds_id][3])

            dig_site_info[ds_id, 0] = ds_x
            dig_site_info[ds_id, 1] = ds_y
            dig_site_info[ds_id, 2] = ds_val
            dig_site_info[ds_id, 3] = ds_coupling

        # Initialize Rover Positions
        initial_agent_positions = np.zeros((p["n_agents"], 3))  # [X, Y, Theta]

        if p["agent_config_type"] == "Random":
            initial_agent_positions = agent_pos_random(dig_site_info)
        elif p["agent_config_type"] == "Concentrated":
            initial_agent_positions = agent_pos_center_concentrated()
        elif p["agent_config_type"] == "Fixed":
            initial_agent_positions = agent_pos_fixed_middle()

        save_agent_configuration(initial_agent_positions, cf_id)


if __name__ == '__main__':
    """
    Create new world configuration files for Dig Sites and rovers
    """
    if p["world_setup"] == "Agent_Only":
        create_agent_setup_only()
    else:
        coupling = 1  # Default coupling requirement for Dig Sites
        create_world_setup(coupling)

    cd = ConstructionDomain()  # Number of Dig Sites, Number of Rovers
    cd.load_world()
    for cf_id in range(p["n_configurations"]):
        cd.reset_world(cf_id)
        agent_path = np.zeros((p["stat_runs"], p["n_agents"], p["steps"], 3))
        for agent_id in range(p["n_agents"]):
            for t in range(p["steps"]):
                agent_path[0:p["stat_runs"], agent_id, t, 0] = cd.agent_configurations[agent_id][cf_id][0]
                agent_path[0:p["stat_runs"], agent_id, t, 1] = cd.agent_configurations[agent_id][cf_id][1]
                agent_path[0:p["stat_runs"], agent_id, t, 2] = cd.agent_configurations[agent_id][cf_id][2]

        create_pickle_file(agent_path, "./Output_Data/", f"Agent_Paths{cf_id}")
        run_visualizer(cf_id=cf_id)
