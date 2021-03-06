# Run benchmark manually

**Objectives** - By following this tutorial, you will be able to:

- generate synthetic data for running lightgbm
- run lightgbm training and inferencing scripts to measure wall time

**Requirements** - To enjoy this tutorial, you need to have installed python dependencies locally (see [instructions](../run/install.md)).

## Generate synthetic data

To generate a synthetic dataset based on sklearn:

=== "Bash"

    ``` bash
    python src/scripts/data_processing/generate_data/generate.py \
        --train_samples 30000 \
        --test_samples 3000 \
        --inferencing_samples 30000 \
        --n_features 4000 \
        --n_informative 400 \
        --random_state 5 \
        --output_train ./data/synthetic/train/ \
        --output_test ./data/synthetic/test/ \
        --output_inference ./data/synthetic/inference/ \
        --type regression
    ```

=== "Powershell"

    ``` powershell
    python src/scripts/data_processing/generate_data/generate.py `
        --train_samples 30000 `
        --test_samples 3000 `
        --inferencing_samples 30000 `
        --n_features 4000 `
        --n_informative 400 `
        --random_state 5 `
        --output_train ./data/synthetic/train/ `
        --output_test ./data/synthetic/test/ `
        --output_inference ./data/synthetic/inference/ `
        --type regression
    ```


!!! note
    Running the synthetic data generation script with these parameter values requires at least 4 GB of RAM available and generates a 754 MB training, a 75 MB testing, and a 744 MB inferencing dataset.

## Run training on synthetic data

=== "Bash"

    ``` bash
    python src/scripts/training/lightgbm_python/train.py \
        --train ./data/synthetic/train/ \
        --test ./data/synthetic/test/ \
        --export_model ./data/models/synthetic-100trees-4000cols/ \
        --objective regression \
        --boosting_type gbdt \
        --tree_learner serial \
        --metric rmse \
        --num_trees 100 \
        --num_leaves 100 \
        --min_data_in_leaf 400 \
        --learning_rate 0.3 \
        --max_bin 16 \
        --feature_fraction 0.15 \
        --device_type cpu
    ```

=== "Powershell"

    ``` powershell
    python src/scripts/training/lightgbm_python/train.py `
        --train ./data/synthetic/train/ `
        --test ./data/synthetic/test/ `
        --export_model ./data/models/synthetic-100trees-4000cols/ `
        --objective regression `
        --boosting_type gbdt `
        --tree_learner serial `
        --metric rmse `
        --num_trees 100 `
        --num_leaves 100 `
        --min_data_in_leaf 400 `
        --learning_rate 0.3 `
        --max_bin 16 `
        --feature_fraction 0.15 `
        --device_type cpu
    ```

!!! note
    `--device_type cpu` is optional here, if you're running on gpu you can use `--device_type gpu` instead.

## Run inferencing on synthetic data (lightgbm python)

=== "Bash"

    ```bash
    python src/scripts/inferencing/lightgbm_python/score.py \
        --data ./data/synthetic/inference/ \
        --model ./data/models/synthetic-100trees-4000cols/ \
        --output ./data/outputs/predictions/ \
        --num_threads 1
    ```

=== "Powershell"

    ``` powershell
    python src/scripts/inferencing/lightgbm_python/score.py `
        --data ./data/synthetic/inference/ `
        --model ./data/models/synthetic-100trees-4000cols/ `
        --output ./data/outputs/predictions/ `
        --num_threads 1
    ```
