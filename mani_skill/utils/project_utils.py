import sys
import os


def find_project_root(current_path, root_folder_name):
    """
    Traverse the directory structure upwards until the specified root folder is found.

    :param current_path: The starting path to begin the search from.
    :param root_folder_name: The name of the root folder to search for.
    :return: The absolute path to the root folder if found, otherwise None.
    """
    while True:
        if os.path.basename(current_path) == root_folder_name:
            return current_path
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            # Reached the filesystem root without finding the folder
            return None
        current_path = parent_path
