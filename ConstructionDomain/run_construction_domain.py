from EvolutionaryAlgorithms.ccea import CCEA
from NeuralNetworks.neural_network import NeuralNetwork
from ConstructionDomainCore.reward_functions import calc_difference, calc_dpp
from ConstructionDomainCore.construction_domain import ConstructionDomain
import numpy as np
from parameters import parameters as p
from global_functions import create_csv_file, save_best_policies, create_pickle_file


def sample_best_team(cd, pops, networks):
    """
    Sample the performance of the team comprised of the best individuals discovered so far during the learning process
    """
    # Select network weights
    for ag in cd.agents:
        policy_id = np.argmax(pops[f'EA{cd.agents[ag].agent_id}'].fitness)
        weights = pops[f'EA{cd.agents[ag].agent_id}'].population[f'pol{policy_id}']
        networks[f'NN{cd.agents[ag].agent_id}'].get_weights(weights)

    g_reward = 0
    for cf_id in range(p["n_configurations"]):
        # Reset rovers to configuration initial conditions
        cd.reset_world(cf_id)
        ds_rewards = np.zeros((p["n_dig_sites"], p["steps"]))
        for step_id in range(p["steps"]):
            # Get rover actions from neural network
            agent_actions = []
            for ag in cd.agents:
                action = networks[f'NN{cd.agents[ag].agent_id}'].run_agent_nn(cd.agents[ag].observations)
                agent_actions.append(action)

            step_rewards = cd.step(agent_actions)
            # Calculate rewards at current time step
            for ds_id in range(p["n_dig_sites"]):
                ds_rewards[ds_id, step_id] = step_rewards[ds_id]

        for p_reward in ds_rewards:
            g_reward += max(p_reward)

    g_reward /= p["n_configurations"]  # Average across configurations

    return g_reward


def cd_global():
    """
    Train agents in the construction domain using the global reward
    """
    # World Setup
    cd = ConstructionDomain()
    cd.load_world()

    # Create dictionary for CCEA populations
    pops = {}
    networks = {}
    for agent_id in range(p["n_agents"]):
        pops[f'EA{agent_id}'] = CCEA(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])
        networks[f'NN{agent_id}'] = NeuralNetwork(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])

    # Perform runs
    srun = p["starting_srun"]
    calls_to_g = {"gen_calls": [], "srun_calls": []}  # Total calls to g
    while srun < p["stat_runs"]:
        print("Run: %i" % srun)
        srun_g_count = 0
        calls_to_g_gen = []
        # Create new CCEA populations
        for pkey in pops:
            pops[pkey].create_new_population()

        reward_history = []
        for gen in range(p["generations"]):
            gen_g_count = 0
            for pkey in pops:
                pops[pkey].select_policy_teams()
                pops[pkey].reset_fitness()

            # Test each team from CCEA
            for team_number in range(p["pop_size"]):
                # Select network weights

                for ag in cd.agents:
                    policy_id = int(pops[f'EA{cd.agents[ag].agent_id}'].team_selection[team_number])
                    weights = pops[f'EA{cd.agents[ag].agent_id}'].population[f'pol{policy_id}']
                    networks[f'NN{cd.agents[ag].agent_id}'].get_weights(weights)

                for cf_id in range(p["n_configurations"]):
                    # Reset environment to configuration initial conditions
                    cd.reset_world(cf_id)
                    ds_rewards = np.zeros((p["n_dig_sites"], p["steps"]))
                    for step_id in range(p["steps"]):
                        # Get rover actions from neural network
                        agent_actions = []
                        for ag in cd.agents:
                            action = networks[f'NN{cd.agents[ag].agent_id}'].run_agent_nn(cd.agents[ag].observations)
                            agent_actions.append(action)

                        step_rewards = cd.step(agent_actions)
                        gen_g_count += 1  # Increment calls to g counter
                        # Calculate rewards at current time step
                        for ds_id in range(p["n_dig_sites"]):
                            ds_rewards[ds_id, step_id] = step_rewards[ds_id]

                    # Update fitness of policies using reward information
                    g_reward = 0
                    for p_reward in ds_rewards:
                         g_reward += max(p_reward)
                    for rover_id in range(p["n_rovers"]):
                        policy_id = int(pops[f'EA{rover_id}'].team_selection[team_number])
                        pops[f'EA{rover_id}'].fitness[policy_id] += g_reward

                # Average reward across number of configurations
                for agent_id in range(p["n_agents"]):
                    policy_id = int(pops[f'EA{agent_id}'].team_selection[team_number])
                    pops[f'EA{agent_id}'].fitness[policy_id] /= p["n_configurations"]

            # Testing Phase (test best agent team found so far) ------------------------------------------------------
            if gen % p["sample_rate"] == 0 or gen == p["generations"] - 1:
                reward_history.append(sample_best_team(cd, pops, networks))
            # --------------------------------------------------------------------------------------------------------
            # Choose new parents and create new offspring population
            for pkey in pops:
                pops[pkey].down_select()
            srun_g_count += gen_g_count
            calls_to_g_gen.append(gen_g_count)

        # Record Output Files
        create_csv_file(reward_history, "Output_Data/", "Global_Reward.csv")
        for agent_id in range(p["n_agents"]):
            best_policy_id = np.argmax(pops[f'EA{agent_id}'].fitness)
            weights = pops[f'EA{agent_id}'].population[f'pol{best_policy_id}']
            save_best_policies(weights, srun, f'RoverWeights{agent_id}', agent_id)

        srun += 1
        calls_to_g["srun_calls"].append(srun_g_count)
        calls_to_g["gen_calls"].append(calls_to_g_gen)

    create_pickle_file(calls_to_g, "Output_Data/", "G_Calls_G")


def cd_difference():
    """
    Train agents using difference reward function
    """
    # World Setup
    cd = ConstructionDomain()
    cd.load_world()

    # Create dictionary for CCEA populations
    pops = {}
    networks = {}
    for agent_id in range(p["n_agents"]):
        pops[f'EA{agent_id}'] = CCEA(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])
        networks[f'NN{agent_id}'] = NeuralNetwork(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])

    # Perform runs
    calls_to_g = {"gen_calls": [], "srun_calls": []}  # Total calls to g
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:
        print("Run: %i" % srun)
        srun_g_count = 0
        calls_to_g_gen = []
        # Create new CCEA populations
        for pkey in pops:
            pops[pkey].create_new_population()

        reward_history = []
        for gen in range(p["generations"]):
            gen_g_count = 0
            for pkey in pops:
                pops[pkey].select_policy_teams()
                pops[pkey].reset_fitness()

            # Each policy in CCEA is tested in randomly selected teams
            for team_number in range(p["pop_size"]):
                # Select network weights
                for ag in cd.agents:
                    policy_id = int(pops[f'EA{cd.agents[ag].agent_id}'].team_selection[team_number])
                    weights = pops[f'EA{cd.agents[ag].agent_id}'].population[f'pol{policy_id}']
                    networks[f'NN{cd.agents[ag].agent_id}'].get_weights(weights)

                for cf_id in range(p["n_configurations"]):
                    # Reset environment to configuration initial conditions
                    cd.reset_world(cf_id)
                    ds_rewards = np.zeros((p["n_dig_sites"], p["steps"]))
                    for step_id in range(p["steps"]):
                        # Get agent actions from neural network
                        agent_actions = []
                        for ag in cd.agents:
                            action = networks[f'NN{cd.agents[ag].agent_id}'].run_agent_nn(cd.agents[ag].observations)
                            agent_actions.append(action)

                        step_rewards = cd.step(agent_actions)
                        gen_g_count += 1  # Increment calls to g counter
                        # Calculate rewards at current time step
                        for ds_id in range(p["n_dig_sites"]):
                            ds_rewards[ds_id, step_id] = step_rewards[ds_id]

                    # Update fitness of policies using reward information
                    g_reward = 0
                    for p_reward in ds_rewards:
                        g_reward += max(p_reward)
                    d_rewards, g_count = calc_difference(cd.dig_sites, cd.agents, g_reward, cd.agent_ds_distances)
                    gen_g_count += g_count  # Increment calls to g counter
                    for agent_id in range(p["n_agents"]):
                        policy_id = int(pops[f'EA{agent_id}'].team_selection[team_number])
                        pops[f'EA{agent_id}'].fitness[policy_id] += d_rewards[agent_id]

                # Average reward across number of configurations
                for agent_id in range(p["n_agents"]):
                    policy_id = int(pops[f'EA{agent_id}'].team_selection[team_number])
                    pops[f'EA{agent_id}'].fitness[policy_id] /= p["n_configurations"]

            # Testing Phase (test best agent team found so far) ------------------------------------------------------
            if gen % p["sample_rate"] == 0 or gen == p["generations"] - 1:
                reward_history.append(sample_best_team(cd, pops, networks))
            # --------------------------------------------------------------------------------------------------------
            # Choose new parents and create new offspring population
            for pkey in pops:
                pops[pkey].down_select()
            srun_g_count += gen_g_count
            calls_to_g_gen.append(gen_g_count)

        # Record Output Files
        create_csv_file(reward_history, "Output_Data/", "Difference_Reward.csv")
        for agent_id in range(p["n_rovers"]):
            best_policy_id = np.argmax(pops[f'EA{agent_id}'].fitness)
            weights = pops[f'EA{agent_id}'].population[f'pol{best_policy_id}']
            save_best_policies(weights, srun, f'RoverWeights{agent_id}', agent_id)

        srun += 1
        calls_to_g["srun_calls"].append(srun_g_count)
        calls_to_g["gen_calls"].append(calls_to_g_gen)

    create_pickle_file(calls_to_g, "Output_Data/", "G_Calls_D")


def cd_dpp():
    """
    Train agents using D++ reward function
    """
    # World Setup
    cd = ConstructionDomain()
    cd.load_world()

    # Create dictionary for CCEA populations
    pops = {}
    networks = {}
    for agent_id in range(p["n_agents"]):
        pops[f'EA{agent_id}'] = CCEA(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])
        networks[f'NN{agent_id}'] = NeuralNetwork(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])

    # Perform runs
    calls_to_g = {"gen_calls": [], "srun_calls": []}  # Total calls to g
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:  # Perform statistical runs
        print("Run: %i" % srun)
        srun_g_count = 0
        calls_to_g_gen = []
        # Create new CCEA populations
        for pkey in pops:
            pops[pkey].create_new_population()

        reward_history = []
        for gen in range(p["generations"]):
            gen_g_count = 0
            for pkey in pops:
                pops[pkey].select_policy_teams()
                pops[pkey].reset_fitness()

            # Each policy in CCEA is tested in randomly selected teams
            for team_number in range(p["pop_size"]):
                # Select network weights
                for ag in cd.agents:
                    policy_id = int(pops[f'EA{cd.agents[ag].agent_id}'].team_selection[team_number])
                    weights = pops[f'EA{cd.agents[ag].agent_id}'].population[f'pol{policy_id}']
                    networks[f'NN{cd.agents[ag].agent_id}'].get_weights(weights)

                for cf_id in range(p["n_configurations"]):
                    # Reset environment to configuration initial conditions
                    cd.reset_world(cf_id)
                    ds_rewards = np.zeros((p["n_dig_sites"], p["steps"]))
                    for step_id in range(p["steps"]):
                        # Get rover actions from neural network
                        rover_actions = []
                        for ag in cd.agents:
                            action = networks[f'NN{cd.agents[ag].agent_id}'].run_agent_nn(cd.agents[ag].observations)
                            rover_actions.append(action)

                        step_rewards = cd.step(rover_actions)
                        gen_g_count += 1  # Increment calls to g counter
                        # Calculate rewards at current time step
                        for ds_id in range(p["n_dig_sites"]):
                            ds_rewards[ds_id, step_id] = step_rewards[ds_id]

                    # Update fitness of policies using reward information
                    g_reward = 0
                    for p_reward in ds_rewards:
                        g_reward += max(p_reward)
                    dpp_rewards, g_count = calc_dpp(cd.dig_sites, cd.agents, g_reward, cd.agent_ds_distances)
                    gen_g_count += g_count  # Increment calls to g counter
                    for agent_id in range(p["n_agents"]):
                        policy_id = int(pops[f'EA{agent_id}'].team_selection[team_number])
                        pops[f'EA{agent_id}'].fitness[policy_id] += dpp_rewards[agent_id]

                # Average reward across number of configurations
                for agent_id in range(p["n_agents"]):
                    policy_id = int(pops[f'EA{agent_id}'].team_selection[team_number])
                    pops[f'EA{agent_id}'].fitness[policy_id] /= p["n_configurations"]

            # Testing Phase (test best agent team found so far) ------------------------------------------------------
            if gen % p["sample_rate"] == 0 or gen == p["generations"] - 1:
                reward_history.append(sample_best_team(cd, pops, networks))
            # --------------------------------------------------------------------------------------------------------
            # Choose new parents and create new offspring population
            for pkey in pops:
                pops[pkey].down_select()
            srun_g_count += gen_g_count
            calls_to_g_gen.append(gen_g_count)

        # Record Output Files
        create_csv_file(reward_history, "Output_Data/", "DPP_Reward.csv")
        for agent_id in range(p["n_agents"]):
            best_policy_id = np.argmax(pops[f'EA{agent_id}'].fitness)
            weights = pops[f'EA{agent_id}'].population[f'pol{best_policy_id}']
            save_best_policies(weights, srun, f'RoverWeights{agent_id}', agent_id)

        srun += 1
        calls_to_g["srun_calls"].append(srun_g_count)
        calls_to_g["gen_calls"].append(calls_to_g_gen)

    create_pickle_file(calls_to_g, "Output_Data/", "G_Calls_DPP")
