"""
Helper math functions
"""
import os
import argparse
import logging
import numpy as np

def bootstrap_ci(data, iterations=1000, operators={'mean':np.mean}, confidence_level=0.95):
    """
    Args:
        data (np.array) : input data
        iterations (int) : how many bootstrapped samples to generate
        operators (Dict[str->func]) : map of functions to produce CI for
        confidence_level (float) : confidence_level = 1-alpha
    
    Returns:
        operators_ci: Dict[str->tuple]
    """
    # values will be stored in a dict
    bootstrap_runs = {}
    for operator_key in operators.keys():
        bootstrap_runs[operator_key] = []

    sample_size = len(data)
    for _ in range(iterations):
        bootstrap = np.random.choice(data, size=sample_size, replace=True)
        for operator_key, operator_func in operators.items():
            bootstrap_runs[operator_key].append(operator_func(bootstrap))

    operators_ci = {}
    for operator_key in operators.keys():
        values = np.array(bootstrap_runs[operator_key])
        ci_left = np.percentile(values, ((1-confidence_level)/2*100))
        ci_right = np.percentile(values, (100-(1-confidence_level)/2*100))
        ci_mean = np.mean(values) # just for fun
        operators_ci[operator_key] = (ci_left, ci_mean, ci_right)

    return(operators_ci)

if __name__ == "__main__":
    sample_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 5.0])
    print(bootstrap_ci(
        sample_data,
        iterations=1000,
        operators={
            'mean':np.mean,
            'p90': (lambda x : np.percentile(x, 90)),
            'p99': (lambda x : np.percentile(x, 99)),
        }
    ))