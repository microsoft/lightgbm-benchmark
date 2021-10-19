import os
import argparse
import logging
import numpy as np

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
    """ Scans input path and returns a list of files.
    
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


class PartitioningEngine():
    """ Class handles partitioning files into chunks with various strategies. """
    PARTITION_MODES = [
        'chunk',
        'roundrobin',
        'append'
    ]

    def __init__(self, mode, number, logger=None):
        self.mode = mode
        self.number = number
        self.logger = logger or logging.getLogger(__name__)

    def split_by_append(self, input_files, output_path, file_count_target):
        """Just appends N++ files in N groups"""
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
        """The function that partition data by size"""
        current_partition_size = 0
        current_partition_index = 0
        self.logger.info(f"Creating partition {current_partition_index}")
        for input_file in input_files:
            self.logger.info(f"Opening input file {input_file}")
            with open(input_file, "r", encoding="utf-8") as input_handler:
                for line in input_handler:
                    if partition_size > 0 and current_partition_size >= partition_size:
                        current_partition_index += 1
                        current_partition_size = 0
                        self.logger.info(f"Creating partition {current_partition_index}")
                    with open(os.path.join(output_path, "part_{:06d}".format(current_partition_index)), 'a', encoding="utf-8") as output_handler:
                        output_handler.write(line)
                        current_partition_size += 1
        self.logger.info(f"Created {current_partition_index+1} partitions")

    def split_by_count(self, input_files, output_path, partition_count):
        """The function that partition data by count"""
        self.logger.info(f"Creating {partition_count} partitions using round robin.")

        partition_files = [open(os.path.join(output_path, "part_{:06d}".format(i)), "w", encoding="utf-8") for i in range(partition_count)]

        current_index = 0
        for input_file in input_files:
            self.logger.info(f"Opening input file {input_file}")
            with open(input_file, "r", encoding="utf-8") as input_handler:
                for line in input_handler:
                    partition_files[current_index % partition_count].write(line)
                    current_index += 1

        for handler in partition_files:
            handler.close()
        self.logger.info(f"Created {partition_count} partitions")

    def run(self, input_path, output_path):
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


class DataBatch():
    # taken from https://datascience.stackexchange.com/questions/47623/how-feed-a-numpy-array-in-batches-in-keras
    def __init__(self, x, y=None, batch_size=0):
        self.x = x
        self.y = y
        if batch_size == 0:
            self.batch_size = x.shape[0]
            self.num_batches = 1
        else:
            self.batch_size = batch_size
            self.num_batches = np.ceil(x.shape[0] / batch_size)
        
        self.batch_idx = np.array_split(range(x.shape[0]), self.num_batches)
        logging.getLogger(__name__).info(f"Creating data batch with {self.num_batches} batches")
    
    def __len__(self):
        return len(self.batch_idx)
    
    def __getitem__(self, idx):
        return self.x[self.batch_idx[idx]], (self.y[self.batch_idx[idx]] if self.y is not None else None)


def numpy_data_load(path, delimiter=","):
    """Loads data using numpy (csv).
    
    Args:
        path (str): path to data file
    Returns:
        numpy_array, number_of_rows (int), number of cols (int)
    """
    raw_data = np.loadtxt(path, delimiter=delimiter)

    return raw_data, raw_data.shape[0], raw_data.shape[1]

def libsvm_data_load(path):
    """Loads data using libsvm.
    
    Args:
        path (str): path to data file
    Returns:
        (y, x), number_of_rows (int), number of cols (int)
    """
    # importing at last minute intentionally
    from sklearn.datasets import load_svmlight_file
    
    x, y = load_svmlight_file(path)

    return (x,y), x.shape[0], x.shape[1]
