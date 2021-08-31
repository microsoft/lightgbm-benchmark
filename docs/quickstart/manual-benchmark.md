# Run benchmark manually

**Objectives** - By following this tutorial, you will be able to:

- generate synthetic data for running lightgbm
- run lightgbm training and inferencing scripts to measure wall time

**Requirements** - To enjoy this tutorial, you need to have installed python dependencies locally (see [instructions](../quickstart/install.md)).

## Generate synthetic data

To generate a synthetic dataset based on sklearn:

```sh
python src/scripts/generate_data/generate.py \
    --train_samples 100000 \
    --test_samples 10000 \
    --inferencing_samples 100000 \
    --n_features 4000 \
    --n_informative 400 \
    --random_state 5 \
    --output_train ./data/synthetic/train/ \
    --output_test ./data/synthetic/test/ \
    --output_inference ./data/synthetic/inference/ \
    --type regression
```

## Run training on synthetic data

```sh
python src/scripts/lightgbm_python/train.py \
    --train ./data/synthetic/train/ \
    --test ./data/synthetic/test/ \
    --export_model ./data/models/synthetic-1200/ \
    --objective regression \
    --boosting_type gbdt \
    --tree_learner serial \
    --metric rmse \
    --num_trees 1200 \
    --num_leaves 100 \
    --min_data_in_leaf 400 \
    --learning_rate 0.3 \
    --max_bin 16 \
    --feature_fraction 0.15
```

## Run inferencing on synthetic data

```sh
python src/scripts/lightgbm_python/score.py \
    --data ./data/synthetic/inference/ \
    --model ./data/models/synthetic-1200/ \
    --output ./data/outputs/predictions/
```

```sh
python src/scripts/lightgbm_cli/score.py \
    --lightgbm_exec ./build/windows/x64/Release/lightgbm.exe \
    --data ./data/synthetic/inference/ \
    --model ./data/models/synthetic-1200/ \
    --output ./data/outputs/predictions/
```
