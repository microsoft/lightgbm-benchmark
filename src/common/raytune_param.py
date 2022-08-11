# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Parses Ray Tune parameters from text arguments (cli or yaml)
"""
import re
import argparse
import logging
from azureml.core import Workspace, Datastore, Dataset
import ray
from ray import tune
from ray.tune import (
    uniform,
    quniform,
    loguniform,
    qloguniform,
    randn,
    qrandn,
    randint,
    qrandint,
    lograndint,
    qlograndint,
    choice,
)


class RayTuneParameterParser():

    # TODO: allow more distributions
    DISTRIBUTIONS_MAP = {"choice": choice,
                         "uniform": uniform,
                         "quniform": quniform,
                         "randint": randint,
                         "qrandint": qrandint,
                         }

    def __init__(self, tunable_parameters):
        self.tunable_parameters = tunable_parameters
        self.parser = None
        self.args = None
        self.unknown_args = None
        self.tunable_params = {}
        self.fixed_params = {}
        self.logger = logging.getLogger(__name__)

    def get_arg_parser(self, parser=None):
        """Adds component/module arguments to a given argument parser.

        Args:
            parser (argparse.ArgumentParser): an argument parser instance

        Returns:
            ArgumentParser: the argument parser instance

        Notes:
            if parser is None, creates a new parser instance
        """
        # add arguments that are specific to the module
        if parser is None:
            parser = argparse.ArgumentParser(__doc__)

        for key in self.tunable_parameters:
            parser.add_argument(
                self.cli_prefix + key,
                type=str,
                required=False,
                default=None,
                help="TODO"
            )

        self.parser = parser
        return self.parser

    def parse_number(self, s):
        return float(s) if '.' in s else int(s)

    def parse_distribution_kwargs(self, param_str):
        str_segs = param_str.split("=")
        return str_segs[0].strip(), self.parse_number(str_segs[1].strip())

    def build_distribution_object(self, param_content):
        param_str = param_content.strip()
        segs = param_str.split("(", 1)
        distribution = segs[0].strip().lower()
        if not distribution in RayTuneParameterParser.DISTRIBUTIONS_MAP:
            raise NotImplementedError(
                "The distribution of %s is not supported." % (distribution))
        distribution_func = RayTuneParameterParser.DISTRIBUTIONS_MAP[distribution]

        if distribution == "choice":
            distribution_params = segs[1].rsplit(")", 1)[0]
            # get the list for the categories of choice
            range_str = distribution_params.split('=')[-1]
            # convert str to list
            distribution_args = eval(range_str)
            return distribution_func(distribution_args)
        else:
            distribution_params = segs[1].rsplit(")", 1)[0].split(",")
            distribution_args = [self.parse_number(param_str.strip(
            )) for param_str in distribution_params if not '=' in param_str and len(param_str.strip()) > 0]
            distribution_kwargs_pairs = [self.parse_distribution_kwargs(
                param_str) for param_str in distribution_params if '=' in param_str]
            distribution_kwargs = {k: v for k, v in distribution_kwargs_pairs}
            return distribution_func(*distribution_args, ** distribution_kwargs)

    # get a list of parameters with user specified search space
    def get_tune_group(self, parameters):
        tune_group = []
        for k in self.tunable_parameters:
            if self.is_tunable(parameters[k]):
                tune_group.append(k)
        return tune_group

    def is_tunable(self, v):
        if isinstance(v, str):
            rlt = '(' in v and v.strip().split(
                "(", 1)[0] in RayTuneParameterParser.DISTRIBUTIONS_MAP
        else:
            rlt = False
        return rlt

    def parse_from_dict(self, parameter_dict):
        """Parses parameters provided in a dictionary to check if they are sweepable.

        Args:
            parameter_dict (dict)

        Returns:
            tunable_params (dict): all sweep parameters from parameter_dict, constructed as sweep sdk objects
            fixed_params (dict): all fixed / constant parameters from parameter_dict
        """
        self.logger.debug(
            f"parsing sweep params from input dict {parameter_dict}")

        self.tunable_params = {}
        self.fixed_params = {}

        self.tune_group = self.get_tune_group(parameter_dict)
        self.tunable_params = {k: self.build_distribution_object(
            parameter_dict[k]) for k in self.tune_group}
        for k in parameter_dict:
            if k in self.tunable_parameters:
                if k not in self.tune_group:
                    self.fixed_params.update(
                        {k: self.parse_number(parameter_dict[k])})
            else:
                self.fixed_params.update({k: parameter_dict[k]})

        self.logger.debug(f"found tunable/sweep params: {self.tunable_params}")
        self.logger.debug(f"found fixed/const params: {self.fixed_params}")
        return self.tunable_params, self.fixed_params

    def parse_from_argparse(self, args):
        """Parses parameters provided as an argparse namespace.

        Args:
            args (argparse.Namespace)

        Returns:
            tunable_params (dict): all sweep parameters from parameter_dict, constructed as sweep sdk objects
            fixed_params (dict): all fixed / constant parameters from parameter_dict
        """
        self.logger.debug(
            f"parsing sweep params from argparse namespace {args}")
        return self.parse_from_dict(vars(args))

    def get_tunable_params(self):
        """ Returns sweep params parsed from last call to parse function. """
        return self.tunable_params

    def get_fixed_params(self):
        """ Returns fixed params parsed from last call to parse function. """
        return self.fixed_params
