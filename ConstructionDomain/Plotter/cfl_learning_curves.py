import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
from plots_common_functions import import_reward_data, get_standard_err_learning
import sys


def generate_cfl_learning_curves(generations, sample_rate, sruns, n_dig_sites, coupling):

    # Plot Color Palette
    color1 = np.array([26, 133, 255]) / 255  # Blue
    color2 = np.array([255, 194, 10]) / 255  # Yellow
    color3 = np.array([230, 97, 0]) / 255  # Orange
    color4 = np.array([93, 58, 155]) / 255  # Purple
    color5 = np.array([211, 95, 183]) / 255  # Fuschia

    # Graph Data
    g_file_path ='../Global/Output_Data/Global_Reward.csv'
    g_reward = import_reward_data(g_file_path, generations, sample_rate, sruns)
    g_stdev = get_standard_err_learning(g_file_path, g_reward, generations, sample_rate, sruns)

    d_file_path = '../Difference/Output_Data/Difference_Reward.csv'
    d_reward = import_reward_data(d_file_path, generations, sample_rate, sruns)
    d_stdev = get_standard_err_learning(d_file_path, d_reward, generations, sample_rate, sruns)

    dpp_file_path = '../D++/Output_Data/DPP_Reward.csv'
    dpp_reward = import_reward_data(dpp_file_path, generations, sample_rate, sruns)
    dpp_stdev = get_standard_err_learning(dpp_file_path, dpp_reward, generations, sample_rate, sruns)

    # cfl_high_file_path = '../CFL_High/Output_Data/CFL_DPP_Rewards.csv'
    # cfl_high_reward = import_reward_data(cfl_high_file_path, generations, sample_rate, sruns)
    # cfl_high_stdev = get_standard_err_learning(cfl_high_file_path, cfl_high_reward, generations, sample_rate, sruns)
    #
    # cfl_file_path = '../CFL3/Output_Data/CFL_DPP_Rewards.csv'
    # cfl_reward = import_reward_data(cfl_file_path, generations, sample_rate, sruns)
    # cfl_stdev = get_standard_err_learning(cfl_file_path, cfl_reward, generations, sample_rate, sruns)
    #
    # cfl_low_path = '../CFL_Low/Output_Data/CFL_DPP_Rewards.csv'
    # cfl_low_reward = import_reward_data(cfl_low_path, generations, sample_rate, sruns)
    # cfl_low_stdev = get_standard_err_learning(cfl_low_path, cfl_low_reward, generations, sample_rate, sruns)

    x_axis = []
    for i in range(generations):
        if i % sample_rate == 0 or i == generations-1:
            x_axis.append(i)
    x_axis = np.array(x_axis)

    # Plot of Data
    plt.plot(x_axis, g_reward, color=color1)
    plt.plot(x_axis, d_reward, color=color2)
    plt.plot(x_axis, dpp_reward, color=color3)
    # plt.plot(x_axis, cfl_high_reward, color=color4)
    # plt.plot(x_axis, cfl_reward, color=color5)
    # plt.plot(x_axis, cfl_low_reward, color='limegreen')

    # Plot of Error
    alpha_val = 0.2
    plt.fill_between(x_axis, g_reward + g_stdev, g_reward - g_stdev, alpha=alpha_val, facecolor=color1)
    plt.fill_between(x_axis, d_reward + d_stdev, d_reward - d_stdev, alpha=alpha_val, facecolor=color2)
    plt.fill_between(x_axis, dpp_reward + dpp_stdev, dpp_reward - dpp_stdev, alpha=alpha_val, facecolor=color3)
    # plt.fill_between(x_axis, cfl_high_reward + cfl_high_stdev, cfl_high_reward - cfl_high_stdev, alpha=alpha_val, facecolor=color4)
    # plt.fill_between(x_axis, cfl_reward + cfl_stdev, cfl_reward - cfl_stdev, alpha=alpha_val, facecolor=color5)
    # plt.fill_between(x_axis, cfl_low_reward + cfl_low_stdev, cfl_low_reward - cfl_low_stdev, alpha=alpha_val, facecolor='limegreen')

    # Graph Details
    plt.xlabel("Generations")
    plt.ylabel("Average Global Reward")
    plt.legend(["Global", "Difference", "D++"], ncol=3, bbox_to_anchor=(0.5, 1.13), loc='upper center', borderaxespad=0)

    # Save the plot
    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig(f"Plots/CFL_{n_dig_sites}DS_C{coupling}.pdf")

    # Show the plot
    plt.show()


if __name__ == '__main__':
    generations = 1000
    sample_rate = 20
    sruns = 30
    n_dig_sites = int(sys.argv[1])
    coupling = int(sys.argv[2])

    generate_cfl_learning_curves(generations, sample_rate, sruns, n_dig_sites, coupling)
