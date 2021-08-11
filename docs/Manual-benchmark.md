# Run benchmark manually

## Python Setup

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or equivalent.

2. Create a conda environment and install dependencies.

    ```ps
    # create conda environment
    conda create --name lightgbmbenchmark python=3.8 -y

    # activate conda environment
    conda activate lightgbmbenchmark

    # install shrike library
    python -m pip install -r requirements.txt
    ```

## Generate synthetic data

```ps
python src/scripts/generate_data/generate.py --n_samples 1000 --n_features 20 --n_informative 5 --n_redundant 0 --random_state 5 --output ./data/synthetic/train.txt --type regression

python src/scripts/generate_data/generate.py --n_samples 1000 --n_features 20 --n_informative 5 --n_redundant 0 --random_state 6 --output ./data/synthetic/test.txt --type regression

python src/scripts/generate_data/generate.py --n_samples 100000 --n_features 20 --n_informative 5 --n_redundant 0 --random_state 7 --output ./data/synthetic/inference.txt --type regression
```

## Run training on synthetic data

```ps
python src/scripts/lightgbm_python/train.py `
    --train ./data/synthetic/train.txt `
    --test ./data/synthetic/test.txt `
    --export_model ./data/models/synthetic.txt `
    --objective regression `
    --boosting_type gbdt `
    --tree_learner serial `
    --metric l2 `
    --num_trees 50 `
    --num_leaves 50 `
    --min_data_in_leaf 1 `
    --learning_rate 0.3 `
    --max_bin 16 `
    --feature_fraction 0.15
```

## Run inferencing on synthetic data

```ps
python src/scripts/lightgbm_python/score.py `
    --data ./data/synthetic/inference.txt `
    --model ./data/models/synthetic.txt
```
