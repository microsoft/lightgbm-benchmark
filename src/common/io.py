import os
import argparse
import logging

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
    raise Exception(f"Provided INPUT path {path} is neither a directory or a file???")


class InputDataLoader():
    """Utility class to load input data with arguments"""
    # current list of supported loaders
    SUPPORTED_LOADERS = ['lightgbm', 'numpy', 'libsvm']

    # prefix used for all argparse
    DEFAULT_ARG_PREFIX = "input_data"

    def __init__(self,
                 allowed_loaders=SUPPORTED_LOADERS,
                 arg_prefix=DEFAULT_ARG_PREFIX,
                 default_loader=None):
        """Initialize data loader.
        Args:
            allowed_loaders (List[str]): list of supported loaders (can restrict to avoid incompatibilities)
            arg_prefix (str): which prefix to use for all argparse
            default_loader (str): name of default loader (if None, will use first in allowed_loaders)
        """
        self.allowed_loaders = allowed_loaders
        self.arg_prefix = arg_prefix
        self.default_loader = default_loader or allowed_loaders[0]
        self.logger = logging.getLogger(__name__)

    def get_arg_parser(self, parser=None):
        """Adds arguments for this class

        Args:
            parser (argparse.ArgumentParser): an argument parser instance

        Returns:
            ArgumentParser: the argument parser instance

        Notes:
            if parser is None, creates a new parser instance
        """
        # add arguments that are specific to the script
        if parser is None:
            parser = argparse.ArgumentParser(__doc__)

        parser.add_argument(f"--{self.arg_prefix}_loader",
            required=False, type=str, default=self.default_loader, choices=self.allowed_loaders, help="use numpy for csv, libsvm for libsvm, or lightgbm for both")
        
        return parser

    def _lightgbm_loader_load(self, path):
        """Loads data using lightgbm construct().
        
        Args:
            path (str): path to data file

        Returns:
            lightgbm_data_reference, number_of_rows (int), number of cols (int)
        """
        self.logger.info(f"Loading {path} with lightgbm")
        # importing at last minute intentionally
        import lightgbm

        data = lightgbm.Dataset(path, free_raw_data=False).construct()
        raw_data = data.get_data()

        self.logger.info(f"Loaded {path} data has {raw_data.num_data()} rows and {raw_data.num_feature()} cols")
        return raw_data, raw_data.num_data(), raw_data.num_feature()

    def _numpy_loader_load(self, path):
        """Loads data using numpy (csv).
        
        Args:
            path (str): path to data file

        Returns:
            numpy_array, number_of_rows (int), number of cols (int)
        """
        self.logger.info(f"Loading {path} with numpy")
        # importing at last minute intentionally
        import numpy

        raw_data = numpy.loadtxt(path, delimiter=",")

        self.logger.info(f"Loaded {path} data has {raw_data.shape[0]} rows and {raw_data.shape[1]} cols")
        return raw_data, raw_data.shape[0], raw_data.shape[1]

    def _libsvm_loader_load(self, path):
        """Loads data using libsvm.
        
        Args:
            path (str): path to data file

        Returns:
            (y, x), number_of_rows (int), number of cols (int)
        """
        self.logger.info(f"Loading {path} with libsvm")
        # importing at last minute intentionally
        from libsvm.svmutil import svm_read_problem

        y, x = svm_read_problem(path, return_scipy=True)

        self.logger.info(f"Loaded {path}, data (X) has {x.shape[0]} rows and {x.shape[1]} cols")
        return (y, x), x.shape[0], x.shape[1]

    def load(self, args, path):
        """Loads data using the right loader"""
        loader = getattr(args, f"{self.arg_prefix}_loader")
        if loader == "lightgbm":
            return self._lightgbm_loader_load(path)
        if loader == "numpy":
            return self._numpy_loader_load(path)
        if loader == "libsvm":
            return self._libsvm_loader_load(path)
        raise NotImplementedError(f"Data loader '{loader}' is not implemented")
