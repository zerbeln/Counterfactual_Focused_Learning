from gridworld import GridWorld
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import sys
import os


def create_plot_gridworld(n_agents, n_targets, width, height):
    gw = GridWorld(width, height)
    gw.load_configuration(n_agents, n_targets)  # Load GridWorld configuration from CSV files

    agent_x = []
    agent_y = []
    for ag in gw.agents:
        agent_x.append(gw.agents[ag].loc[0])
        agent_y.append(gw.agents[ag].loc[1])

    target_x = []
    target_y = []
    for t_loc in gw.targets:
        target_x.append(t_loc[0])
        target_y.append(t_loc[1])

    plt.scatter(target_x, target_y)
    plt.scatter(agent_x, agent_y)
    plt.xlim([-1, width])
    plt.ylim([-1, height])
    plt.legend(["Targets", "Agents"], ncol=2, bbox_to_anchor=(0.5, 1.13), loc='upper center', borderaxespad=0)
    tick = -0.5
    while tick < 8:
        plt.hlines(tick, -1, 8, colors='black')
        plt.vlines(tick, -1, 8, colors='black')
        tick += 1

    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig(f'Plots/{width}x{height}_{n_agents}Gridworld.pdf')

    plt.show()


if __name__ == "__main__":
    n_agents = int(sys.argv[1])
    n_targets = n_agents
    width = int(sys.argv[2])
    height = width
    create_plot_gridworld(n_agents, n_targets, width, height)

