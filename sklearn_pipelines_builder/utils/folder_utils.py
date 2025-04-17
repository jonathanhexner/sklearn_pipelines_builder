import os

def get_unique_folder(base_folder):
    """
    Returns a unique folder path by appending _1, _2, etc. if needed.
    """
    if not os.path.exists(base_folder):
        return base_folder

    counter = 1
    while True:
        new_folder = f"{base_folder}_{counter}"
        if not os.path.exists(new_folder):
            return new_folder
        counter += 1
