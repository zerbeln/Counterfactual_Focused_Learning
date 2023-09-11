from agent import QLearner
import random
import numpy as np
import os
import csv


class GridWorld:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.n_states = height * width
        self.reward = 10
        self.agents = {}  # Dictionary for agent objects
        self.targets = []  # Coordinates of targets in the Gridworld
        self.target_values = None
        self.walls = []  # Coordinates of cells that are walls in the Gridworld

    def assign_target_values(self, n_targets):
        """
        Assign values to each target (these are distance based values)
        """
        self.target_values = np.zeros(n_targets)
        target_distances = np.zeros(n_targets)
        center_x = int(self.width/2)
        center_y = int(self.height/2)
        for t_id, t_loc in enumerate(self.targets):
            x_dist = abs(center_x - t_loc[0])
            y_dist = abs(center_y - t_loc[1])
            total_dist = x_dist + y_dist
            target_distances[t_id] = total_dist

            if total_dist > int(self.width-3):
                self.target_values[t_id] = 10
            else:
                self.target_values[t_id] = 1

        print("Targets Distances: ", target_distances)
        print("Target Values: ", self.target_values)
        print("Total Value: ", sum(self.target_values))

    def create_world(self, n_agents, n_targets):
        for t in range(n_targets):
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)

            # Make sure targets are not placed on top of each other
            while [x, y] in self.targets:
                x = random.randint(0, self.width-1)
                y = random.randint(0, self.height-1)

            self.targets.append([x, y])

        a_loc = []  # Agent locations
        for a in range(n_agents):
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)

            # Make sure agents do not start out on top of other agents or targets
            while [x, y] in self.targets or [x, y] in a_loc:
                x = random.randint(0, self.width-1)
                y = random.randint(0, self.height-1)

            a_loc.append([x, y])
            self.agents[f'A{a}'] = QLearner(self.width*self.height, x, y)

        self.save_configuration()

    def create_center_world(self, n_agents, n_targets, min_dist_center):
        """
        Create Gridworld where agents all start in the center
        """
        center_x = int(self.width/2)
        center_y = int(self.height/2)

        # Four distant targets
        target_distances = []
        for i in range(4):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            dist_to_center = abs(x - center_x) + abs(y - center_y)

            while dist_to_center < min_dist_center or [x, y] in self.targets:
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                dist_to_center = abs(x - center_x) + abs(y - center_y)

            self.targets.append([x, y])
            target_distances.append(dist_to_center)

        # Other targets (low value targets)
        t = 4
        while t < n_targets:
            x = random.randint(2, self.width - 2)
            y = random.randint(2, self.height - 2)
            dist_to_center = abs(x - center_x) + abs(y - center_y)

            # Make sure targets are not placed on top of each other
            while [x, y] in self.targets or dist_to_center >= min_dist_center - 3 or dist_to_center == 0:
                x = random.randint(2, self.width - 2)
                y = random.randint(2, self.height - 2)
                dist_to_center = abs(x - center_x) + abs(y - center_y)

            target_distances.append(dist_to_center)

            self.targets.append([x, y])
            t += 1
        print(target_distances)

        a_loc = []  # Agent locations
        for a in range(n_agents):
            a_loc.append([center_x, center_y])
            self.agents[f'A{a}'] = QLearner(self.width*self.height, center_x, center_y)

        self.save_configuration()

    def save_configuration(self):
        """
        Save the Gridworld configuration to a CSV file
        """
        dir_name = 'World_Config'  # Intended directory for output files

        if not os.path.exists(dir_name):  # If Data directory does not exist, create it
            os.makedirs(dir_name)

        tfile_name = os.path.join(dir_name, f'Target_Config.csv')
        afile_name = os.path.join(dir_name, f'Agent_Config.csv')

        with open(tfile_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for t_loc in self.targets:
                writer.writerow(t_loc)

        with open(afile_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for ag in self.agents:
                writer.writerow(self.agents[ag].loc)

        csvfile.close()

    def load_configuration(self, n_agents, n_targets):
        """
        Load Gridworld configuration from CSV files
        """
        csv_target_input = []
        csv_agent_input = []

        # Load target information
        with open(f'World_Config/Target_Config.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                csv_target_input.append(row)

        for target_id in range(n_targets):
            tx = float(csv_target_input[target_id][0])
            ty = float(csv_target_input[target_id][1])

            self.targets.append([int(tx), int(ty)])

        # Assign values to targets
        self.assign_target_values(n_targets)

        # Load agent information
        with open(f'World_Config/Agent_Config.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                csv_agent_input.append(row)

        for agent_id in range(n_agents):
            ax = float(csv_agent_input[agent_id][0])
            ay = float(csv_agent_input[agent_id][1])

            self.agents[f'A{agent_id}'] = QLearner(self.width*self.height, int(ax), int(ay))

    def check_collision(self, x, y):
        """
        Check for collision with world boundaries
        """
        if x < 0 or x >= self.width:
            return True
        elif y < 0 or y >= self.height:
            return True
        else:
            return False

    def step(self, agent_state, action):
        """
        Agent provides an action, step function provides state-transition and reward
        """
        x, y = agent_state

        # Provide state-transition for agent
        if action == 0:  # Up
            collision = self.check_collision(x, y+1)
            if not collision:
                y += 1
        elif action == 1:  # Down
            collision = self.check_collision(x, y-1)
            if not collision:
                y -= 1
        elif action == 2:  # Left
            collision = self.check_collision(x-1, y)
            if not collision:
                x -= 1
        elif action == 3:  # Right
            collision = self.check_collision(x+1, y)
            if not collision:
                x += 1

        # Return local agent reward and new agent state
        if [x, y] in self.targets:
            t_id = self.targets.index([x, y])
            return self.target_values[t_id], [x, y]
        else:
            return 0, [x, y]

    def calculate_g_reward(self):
        """
        Calculate the global reward for the team of agents
        """
        # Count number of agents at a target
        target_capture_counter = np.zeros(len(self.targets))
        for t_id, t_loc in enumerate(self.targets):
            for ag in self.agents:
                if t_loc == self.agents[ag].loc:
                    target_capture_counter[t_id] += 1

        # Count how many unique targets are captured
        target_values = 0
        for t_id, agent_count in enumerate(target_capture_counter):
            if agent_count > 0:
                target_values += self.target_values[t_id]

        global_reward = (target_values/np.sum(self.target_values))*100

        return global_reward
