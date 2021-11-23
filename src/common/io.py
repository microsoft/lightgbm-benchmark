# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This contains helper functions to handle inputs and outputs arguments
in the benchmark scripts. It also provides some automation routine to handle data.
"""
import os
import argparse
import logging

def input_file_path(path):
    """ Argparse type to resolve input path as single file from directory.
    Given input path can be either a file, or a directory.
    If it's a directory, this returns the path to the unique file it contains.

    Args:
        path (str): either file or directory path
    
    Returns:
        str: path to file, or to unique file in directory
    """
    if os.path.isfile(path):
        logging.getLogger(__name__).info(f"Found INPUT file {path}")
        return path
    if os.path.isdir(path):
        all_files = os.listdir(path)
        if not all_files:
            raise Exception(f"Could not find any file in specified input directory {path}")
        if len(all_files) > 1:
            raise Exception(f"Found multiple files in input file path {path}, use input_directory_path type instead.")
        logging.getLogger(__name__).info(f"Found INPUT directory {path}, selecting unique file {all_files[0]}")
        return os.path.join(path, all_files[0])
    
    logging.getLogger(__name__).critical(f"Provided INPUT path {path} is neither a directory or a file???")
    return path


def get_all_files(path, fail_on_unknown_type=False):
    """ Scans some input path and returns a list of files.
    
    Args:
        path (str): either a file, or directory path
        fail_on_unknown_type (bool): fails if path is neither a file or a dir?

    Returns:
        List[str]: list of paths contained in path
    """
    # if input path is already a file, return as list
    if os.path.isfile(path):
        logging.getLogger(__name__).info(f"Found INPUT file {path}")
        return [path]

    # if input path is a directory, list all files and return
    if os.path.isdir(path):
        all_files = [ os.path.join(path, entry) for entry in os.listdir(path) ]
        if not all_files:
            raise Exception(f"Could not find any file in specified input directory {path}")
        return all_files

    if fail_on_unknown_type:
        raise FileNotFoundError(f"Provided INPUT path {path} is neither a directory or a file???")
    else:
        logging.getLogger(__name__).critical(f"Provided INPUT path {path} is neither a directory or a file???")

    return path
