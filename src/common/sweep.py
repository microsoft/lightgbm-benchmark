# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Parses Sweep parameters from text arguments (cli or yaml)
"""
import re
import argparse
import logging
from azureml.core import Workspace, Datastore, Dataset
from azureml.train.hyperdrive import (
    choice,
    quniform,
    qloguniform,
    qnormal,
    qlognormal,
    uniform,
    loguniform,
    normal,
    lognormal
)

class SweepParameterParser():
    ALLOWED_DISTRIBUTIONS = {
        "random": ["choice", "uniform", "loguniform", "normal", "lognormal", "quniform"],
        "grid": ["choice"],
        "bayesian": ["choice", "quniform", "uniform"]
    }

    def __init__(self, tunable_parameters, cli_prefix, parameter_sampling):
        self.tunable_parameters = tunable_parameters
        self.cli_prefix = cli_prefix
        self.parameter_sampling = parameter_sampling
        self.parser = None
        self.args = None
        self.unknown_args = None
        self.tunable_params = {}
        self.fixed_params = {}

        if self.parameter_sampling not in SweepParameterParser.ALLOWED_DISTRIBUTIONS:
            raise Exception(f"Sampling distribution {self.parameter_sampling} is not in the list of allowed distributiond {SweepParameterParser.ALLOWED_DISTRIBUTIONS}")

    
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
    
    def parse_from_dict(self, parameter_dict):
        self.tunable_params = {}
        self.fixed_params = {}

        # we're building a loop to test every compatible parsing format
        tunable_parsing_methods = {
            # startswith_test_str, method_to_parse
            "choice": self._parse_choice,
            "uniform": self._parse_uniform
        }

        # for any key in tunable parameters
        for param_key in self.tunable_parameters:
            # let's get the value from the parsed args
            if param_key in parameter_dict:
                param_value = parameter_dict[param_key]
            else:
                continue

            if not isinstance(param_value, str):
                # if we hit a default value in the argparser, let's continue
                continue

            # if that value matches any of the parsing methods
            for sweep_param_key in tunable_parsing_methods:
                if param_value.lower().startswith(sweep_param_key):
                    # run the parsing method for this parameter
                    # and add it to the actually tunable params
                    self.tunable_params[param_key] = tunable_parsing_methods[sweep_param_key](param_value)
                    break
            else:
                # if nothing matches, let's consider this a fixed param
                self.fixed_params[param_key] = self._parse_number(param_value)
        return self.tunable_params, self.fixed_params

    def parse_from_argparse(self, args):
        return self.parse_from_dict(vars(args))

    def get_tunable_params(self):
        return self.tunable_params

    def get_fixed_params(self):
        return self.fixed_params

    """
        BELOW: PARSING ALL SWEEP EXPRESSIONS
        see https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters

        choice,
        quniform,
        qloguniform,
        qnormal,
        qlognormal,
        uniform,
        loguniform,
        normal,
        lognormal
    """
    @classmethod
    def _parse_number(cls, number_as_str):
        """Guesses the type of the argument"""
        number_as_str = number_as_str.strip(" ") # remove space at beginning and end

        for type_class in [int, float]:
            try:
                value = type_class(number_as_str)
                return value
            except:
                pass
        else:
            return number_as_str

    @classmethod
    def _parse_choice(cls, parameter_as_str):
        choice_pattern = r"choice\(([0-9\. ,]+)\)"
        matched_choice = re.match(choice_pattern, parameter_as_str)

        if not matched_choice:
            raise Exception(f"Could not match required format {choice_pattern} in expression {parameter_as_str}")

        choice_entry_pattern = r"[0-9\. ]+"
        choice_list = re.findall(choice_entry_pattern, matched_choice.group(1))
        if not choice_list:
            raise Exception(f"Could not match required list of choices with pattern {choice_entry_pattern} in expression {matched_choice.group(1)}")

        return choice(
                *[cls._parse_number(group) for group in choice_list]
        )

    @classmethod
    def _parse_quniform(cls, parameter_as_str):
        return cls._parse_match(
            parameter_as_str,
            r"quniform\(([0-9\. ]+),([0-9\. ]+),([0-9\. ]+)\)",
            quniform
        )

    @classmethod
    def _parse_qloguniform(cls, parameter_as_str):
        return cls._parse_match(
            parameter_as_str,
            r"qloguniform\(([0-9\. ]+),([0-9\. ]+),([0-9\. ]+)\)",
            qloguniform
        )

    @classmethod
    def _parse_qnormal(cls, parameter_as_str):
        return cls._parse_match(
            parameter_as_str,
            r"qnormal\(([0-9\. ]+),([0-9\. ]+),([0-9\. ]+)\)",
            qnormal
        )

    @classmethod
    def _parse_qlognormal(cls, parameter_as_str):
        return cls._parse_match(
            parameter_as_str,
            r"qlognormal\(([0-9\. ]+),([0-9\. ]+),([0-9\. ]+)\)",
            qlognormal
        )

    @classmethod
    def _parse_uniform(cls, parameter_as_str):
        return cls._parse_match(
            parameter_as_str,
            r"uniform\(([0-9\. ]+),([0-9\. ]+)\)",
            uniform
        )

    @classmethod
    def _parse_loguniform(cls, parameter_as_str):
        return cls._parse_match(
            parameter_as_str,
            r"loguniform\(([0-9\. ]+),([0-9\. ]+)\)",
            loguniform
        )

    @classmethod
    def _parse_normal(cls, parameter_as_str):
        return cls._parse_match(
            parameter_as_str,
            r"normal\(([0-9\. ]+),([0-9\. ]+)\)",
            normal
        )

    @classmethod
    def _parse_lognormal(cls, parameter_as_str):
        return cls._parse_match(
            parameter_as_str,
            r"lognormal\(([0-9\. ]+),([0-9\. ]+)\)",
            lognormal
        )

    @classmethod
    def _parse_match(cls, parameter_as_str, pattern, matched_class):
        matched_param = re.match(pattern, parameter_as_str)

        if matched_param:
            return matched_class(
                *[cls._parse_number(group) for group in matched_param.groups()]
            )
        else:
            raise Exception(f"Could not match required format {pattern} in expression {parameter_as_str}")
