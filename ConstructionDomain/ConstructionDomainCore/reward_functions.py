import numpy as np
import copy
from parameters import parameters as p


# DIFFERENCE REWARD --------------------------------------------------------------------------------------------------
def calc_difference(dig_sites, agents, global_reward, agent_ds_dist):
    """
    Calculate each agent's difference reward for the current episode
    """
    g_count = 0
    difference_rewards = np.zeros(p["n_agents"])
    ds_marked = np.zeros(p["n_dig_sites"])
    ds_excavated = np.zeros(p["n_dig_sites"])

    for agent_id in range(p["n_agents"]):
        counterfactual_global_reward = 0.0
        g_count += 1
        for ds in dig_sites:  # For each Dig Site
            ds_reward = 0.0  # Track best Dig Site reward over all time steps for given Dig Site
            for step in range(p["steps"]):
                agent_distances = copy.deepcopy(agent_ds_dist[dig_sites[ds].ds_id][step])
                agent_distances[agent_id] = 1000.00  # Replace Agent action with counterfactual action

                # Check if required observers within range of Dig Site
                for ag in agents:
                    ag_id = agents[ag].agent_id
                    dist = agent_distances[ag_id]
                    if ag_id != agent_id and dist < p["observation_radius"] and agents[ag].type == "Rover":
                        ds_marked[dig_sites[ds].ds_id] = 1
                    elif ag_id != agent_id and dist < p["excavation_radius"] and agents[ag].type == "Excavator":
                        ds_excavated[dig_sites[ds].ds_id] = 1

                # Calculate reward for given Dig Site at current time step
                if ds_marked[dig_sites[ds].ds_id] and ds_excavated[dig_sites[ds].ds_id]:
                    reward = dig_sites[ds].value
                    if reward > ds_reward:
                        ds_reward = reward

            # Update Counterfactual G
            counterfactual_global_reward += ds_reward

        difference_rewards[agent_id] = global_reward - counterfactual_global_reward

    return difference_rewards, g_count


# D++ REWARD ----------------------------------------------------------------------------------------------------------
def calc_dpp(dig_sites, agents, global_reward, rov_ds_dist):
    """
    Calculate D++ rewards for each agent
    """
    d_rewards, g_count = calc_difference(dig_sites, agents, global_reward, rov_ds_dist)
    rewards = np.zeros(p["n_agents"])  # This is just a temporary reward tracker for iterations of counterfactuals
    dpp_rewards = np.zeros(p["n_agents"])
    ds_marked = np.zeros(p["n_dig_sites"])
    ds_excavated = np.zeros(p["n_dig_sites"])

    # Calculate D++ Reward with (TotalAgents - 1) Counterfactuals
    for agent_id in range(p["n_agents"]):
        counterfactual_global_reward = 0.0
        n_counters = p["n_agents"]-1
        g_count += 1
        for ds in dig_sites:
            ds_reward = 0.0  # Track best Dig Site reward over all time steps for given Dig Site
            for step in range(p["steps"]):
                agent_distances = copy.deepcopy(rov_ds_dist[dig_sites[ds].ds_id][step])
                counterfactual_agents = np.ones(int(n_counters)) * agent_distances[agent_id]
                agent_distances = np.append(agent_distances, counterfactual_agents)

                # Check if required observers within range of Dig Site
                for ag in agents:
                    ag_id = agents[ag].agent_id
                    dist = agent_distances[ag_id]
                    if ag_id != agent_id and dist < p["observation_radius"] and agents[ag].type == "Rover":
                        ds_marked[dig_sites[ds].ds_id] = 1
                    elif ag_id != agent_id and dist < p["excavation_radius"] and agents[ag].type == "Excavator":
                        ds_excavated[dig_sites[ds].ds_id] = 1

                # Calculate reward for given Dig Site at current time step
                if ds_marked[dig_sites[ds].ds_id] and ds_excavated[dig_sites[ds].ds_id]:
                    reward = dig_sites[ds].value
                    if reward > ds_reward:
                        ds_reward = reward

            # Update Counterfactual G
            counterfactual_global_reward += ds_reward

        rewards[agent_id] = (counterfactual_global_reward - global_reward)/n_counters

    for agent_id in range(p["n_agents"]):
        # Compare D++ to D, and iterate through n counterfactuals if D++ > D
        if rewards[agent_id] > d_rewards[agent_id]:
            n_counters = 1
            while n_counters < p["n_agents"]:
                counterfactual_global_reward = 0.0
                g_count += 1
                for ds in dig_sites:
                    ds_reward = 0.0  # Track best Dig Site reward over all time steps for given Dig Site
                    for step in range(p["steps"]):
                        agent_distances = copy.deepcopy(rov_ds_dist[dig_sites[ds].ds_id][step])
                        counterfactual_agents = np.ones(int(n_counters)) * agent_distances[agent_id]
                        agent_distances = np.append(agent_distances, counterfactual_agents)

                        # Check if required observers within range of Dig Site
                        for ag in agents:
                            ag_id = agents[ag].agent_id
                            dist = agent_distances[ag_id]
                            if ag_id != agent_id and dist < p["observation_radius"] and agents[ag].type == "Rover":
                                ds_marked[dig_sites[ds].ds_id] = 1
                            elif ag_id != agent_id and dist < p["excavation_radius"] and agents[ag].type == "Excavator":
                                ds_excavated[dig_sites[ds].ds_id] = 1

                        # Calculate reward for given Dig Site at current time step
                        if ds_marked[dig_sites[ds].ds_id] and ds_excavated[dig_sites[ds].ds_id]:
                            reward = dig_sites[ds].value
                            if reward > ds_reward:
                                ds_reward = reward

                    # Update Counterfactual G
                    counterfactual_global_reward += ds_reward

                # Calculate D++ reward with n counterfactuals added
                temp_dpp = (counterfactual_global_reward - global_reward)/n_counters
                if temp_dpp > rewards[agent_id]:
                    rewards[agent_id] = temp_dpp
                    n_counters = p["n_agents"] + 1  # Stop iterating
                else:
                    n_counters += 1

            dpp_rewards[agent_id] = rewards[agent_id]  # Returns D++ reward for this agent
        else:
            dpp_rewards[agent_id] = d_rewards[agent_id]  # Returns difference reward for this agent

    return dpp_rewards, g_count
