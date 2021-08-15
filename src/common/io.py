import os
import argparse

def input_file_path(path):
    """ Resolve input path from AzureML.
    Given input path can be either a file, or a directory.
    If it's a directory, this returns the path to the unique file it contains.

    Args:
        path (str): either file or directory path
    
    Returns:
        str: path to file, or to unique file in directory
    """
    if os.path.isfile(path):
        print(f"Found INPUT file {path}")
        return path
    if os.path.isdir(path):
        all_files = os.listdir(path)
        if not all_files:
            raise Exception(f"Could not find any file in specified input directory {path}")
        if len(all_files) > 1:
            raise Exception(f"Found multiple files in input file path {path}, use input_directory_path type instead.")
        print(f"Found INPUT directory {path}, selecting unique file {all_files[0]}")
        return os.path.join(path, all_files[0])
    raise Exception(f"Provided INPUT path {path} is neither a directory or a file???")
