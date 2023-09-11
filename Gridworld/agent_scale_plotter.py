import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import sys
import matplotlib.colors as mcolors
import math


def import_pickle_data(file_path):
    """
    Load saved Neural Network policies from pickle file
    """

    data_file = open(file_path, 'rb')
    pickle_data = pickle.load(data_file)
    data_file.close()

    return pickle_data


def calc_stdev(input_data, data_mean, sruns):
    """
    Calculates standard deviation for training data.
    """
    temp_val = 0
    for t_id in range(sruns):
        temp_val += (input_data[t_id][-1] - data_mean) ** 2

    temp_val /= (sruns-1)
    standard_err = math.sqrt(temp_val)/math.sqrt(sruns)

    return standard_err


def create_scaling_plot(n_tests, size, x_axis, sruns):
    # Plot Color Palette
    color1 = np.array([26, 133, 255]) / 255  # Blue
    color2 = np.array([255, 194, 10]) / 255  # Yellow
    color3 = np.array([230, 97, 0]) / 255  # Orange
    color4 = np.array([93, 58, 155]) / 255  # Purple
    color5 = np.array([211, 95, 183]) / 255  # Fuschia

    # Q-Learning Data -------------------------------------------------------------
    ql_rewards = []
    ql_data = []
    ql_err = []

    # Global Reward Data ---------------------------------------------------------
    g_rewards = []
    g_data = []
    g_err = []

    # Difference Reward Data --------------------------------------------------------
    d_rewards = []
    d_data = []
    d_err = []

    # PBRS Data -----------------------------------------------------------------------
    pbrs_rewards = []
    pbrs_data = []
    pbrs_err = []

    # CFL Data ------------------------------------------------------------------------
    cfl_rewards = []
    cfl_data = []
    cfl_err = []

    # DRiP Data ------------------------------------------------------------------------
    drip_rewards = []
    drip_data = []
    drip_err = []

    # CFL-P Data ------------------------------------------------------------------------
    cflp_rewards = []
    cflp_data = []
    cflp_err = []

    # Sort the data
    for na in x_axis:
        ql_data.append(import_pickle_data(f"{na}Agents/Output_Data/QLearning_GReward"))
        g_data.append(import_pickle_data(f"{na}Agents/Output_Data/Global_Rewards"))
        d_data.append(import_pickle_data(f"{na}Agents/Output_Data/Difference_Rewards"))
        pbrs_data.append(import_pickle_data(f"{na}Agents/Output_Data/PBRS_Rewards"))
        cfl_data.append(import_pickle_data(f"{na}Agents/Output_Data/CFL_Rewards"))
        drip_data.append(import_pickle_data(f"{na}Agents/Output_Data/DRIP_Rewards"))
        cflp_data.append(import_pickle_data(f"{na}Agents/Output_Data/CFLP_Rewards"))

    # Average the data
    for i in range(n_tests):
        ql_avg_data = np.mean(ql_data[i], axis=0)
        ql_rewards.append(ql_avg_data[-1])
        g_avg_data = np.mean(g_data[i][:], axis=0)
        g_rewards.append(g_avg_data[-1])
        d_avg_data = np.mean(d_data[i][:], axis=0)
        d_rewards.append(d_avg_data[-1])
        pbrs_avg_data = np.mean(pbrs_data[i][:], axis=0)
        pbrs_rewards.append(pbrs_avg_data[-1])
        cfl_avg_data = np.mean(cfl_data[i][:], axis=0)
        cfl_rewards.append(cfl_avg_data[-1])
        drip_avg_data = np.mean(drip_data[i][:], axis=0)
        drip_rewards.append(drip_avg_data[-1])
        cflp_avg_data = np.mean(cflp_data[i][:], axis=0)
        cflp_rewards.append(cflp_avg_data[-1])

        # Calculate standard error in the mean
        ql_err.append(calc_stdev(ql_data[i], ql_rewards[i], sruns))
        g_err.append(calc_stdev(g_data[i], g_rewards[i], sruns))
        d_err.append(calc_stdev(d_data[i], d_rewards[i], sruns))
        pbrs_err.append(calc_stdev(pbrs_data[i], pbrs_rewards[i], sruns))
        cfl_err.append(calc_stdev(cfl_data[i], cfl_rewards[i], sruns))
        drip_err.append(calc_stdev(drip_data[i], drip_rewards[i], sruns))
        cflp_err.append(calc_stdev(cflp_data[i], cflp_rewards[i], sruns))

    # Make Plot -------------------------------------------------------------------------
    plt.errorbar(x_axis, ql_rewards, ql_err,  marker='s', color='black')
    plt.errorbar(x_axis, g_rewards, g_err, marker='^', color=color2)
    plt.errorbar(x_axis, d_rewards, d_err, marker='x', color=color3)
    plt.errorbar(x_axis, pbrs_rewards, pbrs_err, marker='o', color=color5)
    plt.errorbar(x_axis, cfl_rewards, cfl_err, marker='d', color=color4)
    plt.errorbar(x_axis, drip_rewards, drip_err, marker='x', linestyle='--', color="limegreen")
    plt.errorbar(x_axis, cflp_rewards, cflp_err, marker='d', linestyle='--', color=color1)

    # Graph Details
    plt.xlabel("Number of Agents/Targets")
    plt.ylabel("Average Percentage of Targets Captured")
    plt.ylim([0, 102])
    plt.legend(["Local", "Global", "Difference", "PBRS", "CFL", "DRiP", "CFL-P"], ncol=4, bbox_to_anchor=(0.5, 1.13), loc='upper center', borderaxespad=0)
    plt.axhline(y=100, color='black', linestyle=':')

    # Save the plot
    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig(f'Plots/{size}x{size}_AgentScaling.pdf')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    s_runs = 30
    size = int(sys.argv[1])
    x_axis = []
    if size == 8:
        x_axis = [4, 5, 6, 7, 8, 9, 10, 11, 12]
    elif size == 10 or size == 20:
        x_axis = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    n_tests = len(x_axis)

    create_scaling_plot(n_tests, size, x_axis, s_runs)
