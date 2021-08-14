""" Add src/ to path """
import os
import sys
import logging

LIGHTGBM_BENCHMARK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))

if LIGHTGBM_BENCHMARK_ROOT not in sys.path:
    logging.info(f"Adding {LIGHTGBM_BENCHMARK_ROOT} to path")
    sys.path.append(str(LIGHTGBM_BENCHMARK_ROOT))
