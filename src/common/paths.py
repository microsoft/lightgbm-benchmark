import os

LIGHTGBM_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SCRIPTS_SOURCES_ROOT = os.path.join(LIGHTGBM_REPO_ROOT, 'src')
COMPONENTS_ROOT = os.path.join(LIGHTGBM_REPO_ROOT, 'src', 'scripts')
CONFIG_PATH = os.path.join(LIGHTGBM_REPO_ROOT, 'pipelines', 'azureml', 'conf')
