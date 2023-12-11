import numpy as np
import copy
from ConstructionDomainCore.reward_functions import calc_difference
from parameters import parameters as p


def create_counterfactuals_DPP(n_counterfactuals, counterfactual, rover_dist):
    counterfactual_partners = np.ones(n_counterfactuals)

    if counterfactual == 1:  # Constructive counterfactual
        counterfactual_partners *= rover_dist
        return counterfactual_partners
    else:
        counterfactual_partners *= 100.00  # Null counterfactual
        return counterfactual_partners


def create_counterfactuals_dif(counterfactual, rover_dist):
    if counterfactual == 1:
        counterfactual_action = 1000.00
        return counterfactual_action
    else:
        counterfactual_action = rover_dist
        return counterfactual_action


# CFL - Difference REWARD -------------------------------------------------------------------------------------------
def calc_cfl_difference(dig_sites, global_reward, rov_ds_dist, counterfactuals):
    """
    Calculate D rewards for each rover
    """
    g_count = 0
    cfl_d_rewards = np.zeros(p["n_rovers"])
    for agent_id in range(p["n_rovers"]):
        counterfactual_global_reward = 0.0
        g_count += 1
        for pk in dig_sites:  # For each Dig Site
            ds_reward = 0.0  # Track best Dig Site reward over all time steps for given Dig Site
            for step in range(p["steps"]):
                observer_count = 0
                rover_distances = copy.deepcopy(rov_ds_dist[dig_sites[pk].ds_id][step])
                c_action = create_counterfactuals_dif(counterfactuals, rover_distances[agent_id])
                rover_distances[agent_id] = c_action  # Replace Rover action with counterfactual action
                sorted_distances = np.sort(rover_distances)  # Sort from least to greatest

                # Check if required observers within range of Dig Site
                for i in range(int(dig_sites[pk].coupling)):
                    if sorted_distances[i] < p["observation_radius"]:
                        observer_count += 1

                # Calculate reward for given Dig Site at current time step
                if observer_count >= int(dig_sites[pk].coupling):
                    summed_observer_distances = sum(sorted_distances[0:int(dig_sites[pk].coupling)])
                    reward = dig_sites[pk].value / (summed_observer_distances / dig_sites[pk].coupling)
                    if reward > ds_reward:
                        ds_reward = reward

            # Update Counterfactual G
            counterfactual_global_reward += ds_reward

        cfl_d_rewards[agent_id] = global_reward - counterfactual_global_reward

    return cfl_d_rewards, g_count


# S-D++ REWARD -------------------------------------------------------------------------------------------------------
def calc_cfl_dpp(dig_sites, global_reward, rov_ds_dist, counterfactuals):
    """
    Calculate D++ rewards for each rover
    """
    d_rewards, g_count = calc_difference(dig_sites, global_reward, rov_ds_dist)
    rewards = np.zeros(p["n_rovers"])  # This is just a temporary reward tracker for iterations of counterfactuals
    cfl_dpp_rewards = np.zeros(p["n_rovers"])

    # Calculate D++ Reward with (TotalAgents - 1) Counterfactuals
    for agent_id in range(p["n_rovers"]):
        counterfactual_global_reward = 0.0
        n_counters = p["n_rovers"]-1
        g_count += 1
        for pk in dig_sites:
            ds_reward = 0.0  # Track best Dig Site reward over all time steps for given Dig Site
            for step in range(p["steps"]):
                observer_count = 0
                rover_distances = copy.deepcopy(rov_ds_dist[dig_sites[pk].ds_id][step])
                ctype = counterfactuals[dig_sites[pk].ds_id][agent_id]
                c_partners = create_counterfactuals_DPP(n_counters, ctype, rover_distances[agent_id])
                rover_distances = np.append(rover_distances, c_partners)
                sorted_distances = np.sort(rover_distances)  # Sort from least to greatest

                # Check if required observers within range of Dig Site
                for i in range(int(dig_sites[pk].coupling)):
                    if sorted_distances[i] < p["observation_radius"]:
                        observer_count += 1

                # Calculate reward for given Dig Site at current time step
                if observer_count >= dig_sites[pk].coupling:
                    summed_observer_distances = sum(sorted_distances[0:int(dig_sites[pk].coupling)])
                    reward = dig_sites[pk].value/(summed_observer_distances/dig_sites[pk].coupling)
                    if reward > ds_reward:
                        ds_reward = reward

            # Update Counterfactual G
            counterfactual_global_reward += ds_reward

        rewards[agent_id] = (counterfactual_global_reward - global_reward)/n_counters

    for agent_id in range(p["n_rovers"]):
        # Compare D++ to D, and iterate through n counterfactuals if D++ > D
        if rewards[agent_id] > d_rewards[agent_id]:
            n_counters = 1
            while n_counters < p["n_rovers"]:
                counterfactual_global_reward = 0.0
                g_count += 1
                for pk in dig_sites:
                    observer_count = 0
                    ds_reward = 0.0  # Track best Dig Site reward over all time steps for given Dig Site
                    for step in range(p["steps"]):
                        rover_distances = copy.deepcopy(rov_ds_dist[dig_sites[pk].ds_id][step])
                        ctype = counterfactuals[dig_sites[pk].ds_id][agent_id]
                        c_partners = create_counterfactuals_DPP(n_counters, ctype, rover_distances[agent_id])
                        rover_distances = np.append(rover_distances, c_partners)
                        sorted_distances = np.sort(rover_distances)  # Sort from least to greatest

                        # Check if required observers within range of Dig Site
                        for i in range(int(dig_sites[pk].coupling)):
                            if sorted_distances[i] < p["observation_radius"]:
                                observer_count += 1

                        # Calculate reward for given Dig Site at current time step
                        if observer_count >= dig_sites[pk].coupling:
                            summed_observer_distances = sum(sorted_distances[0:int(dig_sites[pk].coupling)])
                            reward = dig_sites[pk].value/(summed_observer_distances/dig_sites[pk].coupling)
                            if reward > ds_reward:
                                ds_reward = reward

                    # Update Counterfactual G
                    counterfactual_global_reward += ds_reward

                # Calculate D++ reward with n counterfactuals added
                temp_dpp = (counterfactual_global_reward - global_reward)/n_counters
                if temp_dpp > rewards[agent_id]:
                    rewards[agent_id] = temp_dpp
                    n_counters = p["n_rovers"] + 1  # Stop iterating
                else:
                    n_counters += 1

            cfl_dpp_rewards[agent_id] = rewards[agent_id]  # Returns D++ reward for this agent
        else:
            cfl_dpp_rewards[agent_id] = d_rewards[agent_id]  # Returns difference reward for this agent

    return cfl_dpp_rewards, g_count


def cfl_dpp_dif(dig_sites, global_reward, rov_ds_dist, counterfactuals):
    """
    Calculate D++ rewards for each rover
    """
    sd_rewards, g_count = calc_cfl_difference(dig_sites, global_reward, rov_ds_dist, counterfactuals)
    rewards = np.zeros(p["n_rovers"])  # This is just a temporary reward tracker for iterations of counterfactuals
    cfl_dpp_rewards = np.zeros(p["n_rovers"])

    # Calculate D++ Reward with (TotalAgents - 1) Counterfactuals
    for agent_id in range(p["n_rovers"]):
        counterfactual_global_reward = 0.0
        n_counters = p["n_rovers"] - 1
        g_count += 1
        for pk in dig_sites:
            ds_reward = 0.0  # Track best Dig Site reward over all time steps for given Dig Site
            for step in range(p["steps"]):
                observer_count = 0
                rover_distances = copy.deepcopy(rov_ds_dist[dig_sites[pk].ds_id][step])
                ctype = counterfactuals[dig_sites[pk].ds_id][agent_id]
                c_partners = create_counterfactuals_DPP(n_counters, ctype, rover_distances[agent_id])
                rover_distances = np.append(rover_distances, c_partners)
                sorted_distances = np.sort(rover_distances)  # Sort from least to greatest

                # Check if required observers within range of Dig Site
                for i in range(int(dig_sites[pk].coupling)):
                    if sorted_distances[i] < p["observation_radius"]:
                        observer_count += 1

                # Calculate reward for given Dig Site at current time step
                if observer_count >= dig_sites[pk].coupling:
                    summed_observer_distances = sum(sorted_distances[0:int(dig_sites[pk].coupling)])
                    reward = dig_sites[pk].value / (summed_observer_distances / dig_sites[pk].coupling)
                    if reward > ds_reward:
                        ds_reward = reward

            # Update Counterfactual G
            counterfactual_global_reward += ds_reward

        rewards[agent_id] = (counterfactual_global_reward - global_reward) / n_counters

    for agent_id in range(p["n_rovers"]):
        # Compare D++ to D, and iterate through n counterfactuals if D++ > D
        if rewards[agent_id] > sd_rewards[agent_id]:
            n_counters = 1
            while n_counters < p["n_rovers"]:
                counterfactual_global_reward = 0.0
                g_count += 1
                for pk in dig_sites:
                    observer_count = 0
                    ds_reward = 0.0  # Track best Dig Site reward over all time steps for given Dig Site
                    for step in range(p["steps"]):
                        rover_distances = copy.deepcopy(rov_ds_dist[dig_sites[pk].ds_id][step])
                        ctype = counterfactuals[dig_sites[pk].ds_id][agent_id]
                        c_partners = create_counterfactuals_DPP(n_counters, ctype, rover_distances[agent_id])
                        rover_distances = np.append(rover_distances, c_partners)
                        sorted_distances = np.sort(rover_distances)  # Sort from least to greatest

                        # Check if required observers within range of Dig Site
                        for i in range(int(dig_sites[pk].coupling)):
                            if sorted_distances[i] < p["observation_radius"]:
                                observer_count += 1

                        # Calculate reward for given Dig Site at current time step
                        if observer_count >= dig_sites[pk].coupling:
                            summed_observer_distances = sum(sorted_distances[0:int(dig_sites[pk].coupling)])
                            reward = dig_sites[pk].value / (summed_observer_distances / dig_sites[pk].coupling)
                            if reward > ds_reward:
                                ds_reward = reward

                    # Update Counterfactual G
                    counterfactual_global_reward += ds_reward

                # Calculate D++ reward with n counterfactuals added
                temp_dpp = (counterfactual_global_reward - global_reward) / n_counters
                if temp_dpp > rewards[agent_id]:
                    rewards[agent_id] = temp_dpp
                    n_counters = p["n_rovers"] + 1  # Stop iterating
                else:
                    n_counters += 1

            cfl_dpp_rewards[agent_id] = rewards[agent_id]  # Returns D++ reward for this agent
        else:
            cfl_dpp_rewards[agent_id] = sd_rewards[agent_id]  # Returns difference reward for this agent

    return cfl_dpp_rewards, g_count
