import numpy as np


def distance_based(gw):
    """
    Create counterfactuals that agents will use for CFL learning
    """
    counterfactuals = [[0 for i in range(len(gw.targets))] for j in range(len(gw.agents))]
    for a_id, ag in enumerate(gw.agents):
        for t_id, t_loc in enumerate(gw.targets):
            x_dist = abs(gw.agents[ag].initial_position[0] - t_loc[0])
            y_dist = abs(gw.agents[ag].initial_position[1] - t_loc[1])
            total_dist = x_dist + y_dist

            if total_dist > (gw.width - 3):
                counterfactuals[a_id][t_id] = 1

    return counterfactuals


def value_based(gw):
    """
    Counterfactual states reflect target values
    """
    counterfactuals = [[0 for i in range(len(gw.targets))] for j in range(len(gw.agents))]
    for a_id in range(len(gw.agents)):
        for t_id in range(len(gw.targets)):
            if gw.target_values[t_id] > 1:
                counterfactuals[a_id][t_id] = 1

    return counterfactuals


def agent_dist_split(gw, n_agents):
    """
    Create counterfactuals that divide agents evenly between capturing far away targets and close targets
    """
    counterfactuals = [[0 for i in range(len(gw.targets))] for j in range(len(gw.agents))]
    for a_id, ag in enumerate(gw.agents):
        for t_id, t_loc in enumerate(gw.targets):
            x_dist = abs(gw.agents[ag].initial_position[0] - t_loc[0])
            y_dist = abs(gw.agents[ag].initial_position[1] - t_loc[1])
            total_dist = x_dist + y_dist
            if total_dist <= (gw.width - 3) and a_id < n_agents:
                counterfactuals[a_id][t_id] = 1
            elif total_dist > (gw.width - 3) and a_id >= n_agents:
                counterfactuals[a_id][t_id] = 1

        assert(1 in counterfactuals[a_id])
    return counterfactuals


def poi_assignments(gw):
    """
    Assign each agent to a unique POI
    """
    counterfactuals = [[0 for i in range(len(gw.targets))] for j in range(len(gw.agents))]
    for a_id, ag in enumerate(gw.agents):
        for t_id, t_loc in enumerate(gw.targets):
            if a_id == t_id:
                counterfactuals[a_id][t_id] = 1

    return counterfactuals


def create_counterfactuals(gw, ctype, n_agents):
    """
    Generate counterfactual states for agents
    """
    if ctype == "distance":
        counterfactuals = distance_based(gw)
        return counterfactuals
    elif ctype == "split":
        counterfactuals = agent_dist_split(gw, n_agents)
        return counterfactuals
    elif ctype == "assign":
        counterfactuals = poi_assignments(gw)
        return counterfactuals
    else:
        counterfactuals = value_based(gw)
        return counterfactuals


def calc_cfl_difference(g_reward, gw, counterfactuals):
    """
    Calculate the difference reward for each agent using CFL counterfactuals
    """
    difference_reward = np.zeros(len(gw.agents))

    # Count number of agents at a target
    for i in range(len(gw.agents)):
        target_capture_counter = np.zeros(len(gw.targets))
        for t_id, t_loc in enumerate(gw.targets):
            for a_id, ag in enumerate(gw.agents):
                if t_loc == gw.agents[ag].loc and a_id == i and counterfactuals[i][t_id] == 1:
                    pass
                elif t_loc == gw.agents[ag].loc:
                    target_capture_counter[t_id] += 1

        # Count how many targets are captured in counterfactual state
        target_values = 0
        for t_id, agent_count in enumerate(target_capture_counter):
            if agent_count > 0:
                target_values += gw.target_values[t_id]

        counterfactual_global_reward = (target_values/np.sum(gw.target_values))*100
        difference_reward[i] = g_reward - counterfactual_global_reward

    return difference_reward
