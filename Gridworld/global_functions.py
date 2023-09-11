import os
import pickle
import csv


def create_pickle_file(input_data, dir_name, file_name):
    """
    Create a pickle file using provided data in the specified directory
    """

    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    path_name = os.path.join(dir_name, file_name)
    rover_file = open(path_name, 'wb')
    pickle.dump(input_data, rover_file)
    rover_file.close()


def create_csv_file(input_array, dir_name, file_name):
    """
    Save array as a CSV file in the specified directory
    """
    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    save_file_name = os.path.join(dir_name, file_name)
    with open(save_file_name, 'a+', newline='') as csvfile:  # Record reward history for each stat run
        writer = csv.writer(csvfile)
        writer.writerow(input_array)

