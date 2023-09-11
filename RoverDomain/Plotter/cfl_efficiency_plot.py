from plots_common_functions import import_pickle_data
import os
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import sys


def generate_efficiency_plots(generations, sruns, n_poi, coupling, sample_rate=20):
    # Plot Color Palette
    color1 = np.array([26, 133, 255]) / 255  # Blue
    color2 = np.array([255, 194, 10]) / 255  # Yellow
    color3 = np.array([230, 97, 0]) / 255  # Orange
    color4 = np.array([93, 58, 155]) / 255  # Purple
    color5 = np.array([211, 95, 183]) / 255  # Fuschia

    # Graph Data
    g_file_path = '../Global/Output_Data/G_Calls_G'
    g_data = import_pickle_data(g_file_path)  # Data is imported as a dictionary of arrays
    g_gen_eff = np.mean(g_data['gen_calls'], axis=0)
    # g_stdev = get_standard_err_learning(g_file_path, g_reward, generations, sample_rate, sruns)

    d_file_path = '../Difference/Output_Data/G_Calls_D'
    d_data = import_pickle_data(d_file_path)  # Data is imported as a dictionary of arrays
    d_gen_eff = np.mean(d_data['gen_calls'], axis=0)
    # d_stdev = get_standard_err_learning(d_file_path, d_reward, generations, sample_rate, sruns)

    dpp_file_path = '../D++/Output_Data/G_Calls_DPP'
    dpp_data = import_pickle_data(dpp_file_path)  # Data is imported as a dictionary of arrays
    dpp_gen_eff = np.mean(dpp_data['gen_calls'], axis=0)
    # dpp_stdev = get_standard_err_learning(dpp_file_path, dpp_reward, generations, sample_rate, sruns)

    cfl_high_file_path = '../CFL_High/Output_Data/G_Calls_CFL'
    cfl_high_data = import_pickle_data(cfl_high_file_path)  # Data is imported as a dictionary of arrays
    cfl_high_gen_eff = np.mean(cfl_high_data['gen_calls'], axis=0)
    # cfl_high_stdev = get_standard_err_learning(cfl_high_file_path, cfl_high_reward, generations, sample_rate, sruns)

    cfl_file_path = '../CFL3/Output_Data/G_Calls_CFL'
    cfl_data = import_pickle_data(cfl_file_path)  # Data is imported as a dictionary of arrays
    cfl_gen_eff = np.mean(cfl_data['gen_calls'], axis=0)
    # cfl_stdev = get_standard_err_learning(cfl2_file_path, cfl2_reward, generations, sample_rate, sruns)

    cfl_low_path = '../CFL_Low/Output_Data/G_Calls_CFL'
    cfl_low_data = import_pickle_data(cfl_low_path)  # Data is imported as a dictionary of arrays
    cfl_low_gen_eff = np.mean(cfl_low_data['gen_calls'], axis=0)
    # cfl_low_stdev = get_standard_err_learning(cfl_low_path, cfl_low_reward, generations, sample_rate, sruns)

    x_axis = [i for i in range(generations)]
    x_axis = np.array(x_axis)

    # Plot of Data
    # plt.plot(x_axis, g_gen_eff, color=color2)
    # plt.plot(x_axis, d_gen_eff, color=color1)
    plt.plot(x_axis, dpp_gen_eff, color=color3)
    plt.plot(x_axis, cfl_high_gen_eff, color=color4)
    plt.plot(x_axis, cfl_gen_eff, color=color5)
    plt.plot(x_axis, cfl_low_gen_eff, color='limegreen')

    # Graph Details
    plt.xlabel("Generations")
    plt.ylabel("Calls to G")
    plt.legend(["D++", "CFL-High", 'CFL-Team', 'CFL-Low'], ncol=4, bbox_to_anchor=(0.5, 1.075), loc='upper center', borderaxespad=0)

    # Save the plot
    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig(f"Plots/CFL_{n_poi}POI_C{coupling}_Efficiency.pdf")

    plt.show()
    

if __name__ == "__main__":
    generations = 6000
    sruns = 30
    n_poi = int(sys.argv[1])
    coupling = int(sys.argv[2])

    generate_efficiency_plots(generations, sruns, n_poi, coupling)
