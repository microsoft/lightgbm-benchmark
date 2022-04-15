# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This contains helper functions to handle inputs and outputs arguments
in the benchmark scripts. It also provides some automation routine to handle data.
"""
import os
import argparse
import logging
from os.path import exists

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
    # check the existence of the path
    if exists(path) == False: 
        raise Exception(f"The specified path {path} does not exist.")

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


class PartitioningEngine():
    """This class handles partitioning data files into chunks with various strategies. """
    PARTITION_MODES = [
        'chunk',
        'roundrobin',
        'append'
    ]

    def __init__(self, mode, number, header=False, logger=None):
        """Constructs and setup of the engine
        
        Args:
            mode (str): which partition mode (in PartitioningEngine.PARTITION_MODE list)
            number (int): parameter, behavior depends on mode
            header (bool): are there header in the input files?
            logger (logging.logger): a custom logger, if needed, for this engine to log
        """
        self.mode = mode
        self.number = number
        self.header = header
        self.logger = logger or logging.getLogger(__name__)

    def split_by_append(self, input_files, output_path, file_count_target):
        """Just appends N++ files in N groups.
        
        Args:
            input_files (List[str]): list of file paths
            output_path (str): directory path, where to write the partitions
            file_count_target (int): how many partitions we want
        """
        if len(input_files) < file_count_target:
            raise Exception(f"To use mode=append, the number of input files ({len(input_files)}) needs to be higher than requested number of output files ({file_count_target})")

        # each partition starts as an empty list
        partitions = [
            [] for i in range(file_count_target)
        ]

        # loop on all files, and put them in one partition
        for index, input_file in enumerate(input_files):
            partitions[index % file_count_target].append(input_file)
        
        self.logger.info(f"Shuffled {len(input_files)} files into {file_count_target} partitions.")

        # then write each partition by appending content
        for current_partition_index, partition in enumerate(partitions):
            self.logger.info(f"Writing partition {current_partition_index}...")
            with open(os.path.join(output_path, "part_{:06d}".format(current_partition_index)), 'a', encoding="utf-8") as output_handler:
                for input_file in partition:
                    self.logger.info(f"Reading input file {input_file}...")
                    with open(input_file, 'r') as input_handler:
                        output_handler.write(input_handler.read())

        self.logger.info(f"Created {current_partition_index+1} partitions")


    def split_by_size(self, input_files, output_path, partition_size):
        """Splits input files into a variable number of partitions
        by chunking a fixed number of lines from inputs into each
        output file.

        Args:
            input_files (List[str]): list of file paths
            output_path (str): directory path, where to write the partitions
            partition_size (int): how many lines per partition
        """
        current_partition_size = 0
        current_partition_index = 0
        self.logger.info(f"Creating partition {current_partition_index}")

        header_line = None # there can be only on header line

        for input_file in input_files:
            self.logger.info(f"Opening input file {input_file}")
            with open(input_file, "r", encoding="utf-8") as input_handler:
                for line in input_handler:
                    if self.header and header_line is None:
                        # if first line of first input file
                        # write that line in every partition
                        header_line = line

                    if partition_size > 0 and current_partition_size >= partition_size:
                        current_partition_index += 1
                        current_partition_size = 0
                        self.logger.info(f"Creating partition {current_partition_index}")
                        
                    with open(os.path.join(output_path, "part_{:06d}".format(current_partition_index)), 'a', encoding="utf-8") as output_handler:
                        if self.header and current_partition_size == 0:
                            # put header before anything else
                            output_handler.write(header_line)

                        output_handler.write(line)
                        current_partition_size += 1
        self.logger.info(f"Created {current_partition_index+1} partitions")

    def split_by_count(self, input_files, output_path, partition_count):
        """Splits input files into a fixed number of partitions by round-robin
        shuffling of the lines of input files.

        Args:
            input_files (List[str]): list of file paths
            output_path (str): directory path, where to write the partitions
            partition_count (int): how many lines per partition
        """
        self.logger.info(f"Creating {partition_count} partitions using round robin.")

        partition_files = [open(os.path.join(output_path, "part_{:06d}".format(i)), "w", encoding="utf-8") for i in range(partition_count)]

        current_index = 0
        header_line = None # there can be only on header line

        for input_file in input_files:
            self.logger.info(f"Opening input file {input_file}")
            with open(input_file, "r", encoding="utf-8") as input_handler:
                for line_index, line in enumerate(input_handler):
                    if self.header and header_line is None:
                        # if first line of first input file
                        # write that line in every partition
                        header_line = line
                        for partition_file in partition_files:
                            partition_file.write(header_line)
                        continue
                    elif self.header and line_index == 0:
                        # if first line of 2nd... input file, just pass
                        continue

                    partition_files[current_index % partition_count].write(line)
                    current_index += 1

        for handler in partition_files:
            handler.close()
        self.logger.info(f"Created {partition_count} partitions")

    def run(self, input_path, output_path):
        """Runs the partition based on provided arguments.
        
        Args:
            input_path (str): path to input file(s)
            output_path (str): path to store output partitions
        """
        # Retrieve all input files
        if os.path.isfile(input_path):
            self.logger.info("Input is one unique file")
            file_names = [os.path.basename(input_path)]
            input_files = [input_path]
        else:
            self.logger.info("Input is a directory, listing all of them for processing")
            file_names = os.listdir(input_path)
            input_files = [os.path.join(input_path, file) for file in file_names]
            self.logger.info("Found {} files in {}".format(len(input_files), input_path))

        if self.mode == "chunk":
            self.split_by_size(input_files, output_path, self.number)
        elif self.mode == "roundrobin":
            self.split_by_count(input_files, output_path, self.number)
        elif self.mode == "append":
            self.split_by_append(input_files, output_path, self.number)
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented.")
