import pickle
import torch

from mesh_transform import base_path


def retrieve_groundtruth_rotation(scene_nr, image_nr, instance_id):
    folder_name = base_path + 'scene1-10/scene{}/labels/{}_label.pkl'.format(scene_nr, image_nr)
    with open(folder_name, 'rb') as f:
        data = pickle.load(f)
        rot_tens = torch.tensor(data["rotations"][instance_id])
    return rot_tens


def retrieve_groundtruth_translation(scene_nr, image_nr, instance_id):
    folder_name = base_path + 'scene1-10/scene{}/labels/{}_label.pkl'.format(scene_nr, image_nr)
    with open(folder_name, 'rb') as f:
        data = pickle.load(f)
        trans = data["translations"][instance_id]
    return trans


def filter_items(file_path):
    """
    Reads a text file and returns the item names that start with "shoe", "cutlery", or "teapot".

    Parameters:
    file_path (str): The path to the text file.

    Returns:
    list: A list of item names that match the specified criteria.
    """
    # Define the prefixes to filter by
    prefixes = ["teapot", "shoe", "cutlery"]

    # Initialize an empty list to store matching item names
    matching_items = []

    try:
        # Open the file and read it line by line
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line into parts (id, number, item name)
                parts = line.strip().split()

                # Ensure the line has the expected format
                if len(parts) != 3:
                    continue

                item_name = parts[2]

                # Check if the item name starts with any of the specified prefixes
                if any(item_name.startswith(prefix) for prefix in prefixes):
                    matching_items.append(item_name)

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

    return matching_items