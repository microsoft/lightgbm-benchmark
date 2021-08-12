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
python src/scripts/generate_data/generate.py `
    --train_samples 1000 `
    --test_samples 100 `
    --inferencing_samples 10000 `
    --n_features 4000 `
    --n_informative 400 `
    --random_state 5 `
    --output ./data/synthetic/ `
    --type regression
```

## Run training on synthetic data

```ps
python src/scripts/lightgbm_python/train.py `
    --train ./data/synthetic/train.txt `
    --test ./data/synthetic/train.txt `
    --export_model ./data/models/synthetic-1200.txt `
    --objective regression `
    --boosting_type gbdt `
    --tree_learner serial `
    --metric rmse `
    --num_trees 1200 `
    --num_leaves 100 `
    --min_data_in_leaf 400 `
    --learning_rate 0.3 `
    --max_bin 16 `
    --feature_fraction 0.15
```

## Run inferencing on synthetic data

```ps
python src/scripts/lightgbm_python/score.py `
    --data ./data/synthetic/inference.txt `
    --model ./data/models/synthetic-1200.txt `
    --output ./data/outputs/predictions-1200-py.txt
```

```ps
python src/scripts/lightgbm_cli/score.py `
    --lightgbm_exec ./build/windows/x64/Release/lightgbm.exe `
    --data ./data/synthetic/inference.txt `
    --model ./data/models/synthetic-1200.txt `
    --output ./data/outputs/predictions-1200-cli.txt
```
