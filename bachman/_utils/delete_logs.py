"""
Delete all files in a folder.
"""

import os
import shutil


def delete_logs(folder_path: str = "./logs"):
    """
    Delete all files in a folder.
    """

    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    # Iterate over all the files in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if it's a file and delete it
        if os.path.isfile(file_path):
            os.remove(file_path)
            # print(f"Deleted file: {file_path}")
        # If it's a directory, you can choose to delete it as well
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
            # print(f"Deleted directory: {file_path}")


# Example usage
# delete_all_files_in_folder('/path/to/your/folder')
