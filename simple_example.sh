#!/bin/bash

python src/scripts/generate_data/generate.py --train_samples 10000 --test_samples 10000 --inferencing_samples 10000 --n_features 1000 --n_informative 100 --random_state 5 --output_train ./data/synthetic/train/ --output_test ./data/synthetic/test/ --output_inference ./data/synthetic/inference/ --type regression
python src/scripts/lightgbm_python/train.py --train ./data/synthetic/train/ --test ./data/synthetic/test/ --export_model ./data/models/synthetic-1200/ --objective regression --boosting_type gbdt --tree_learner serial --metric rmse --num_trees 300 --num_leaves 100 --min_data_in_leaf 40 --learning_rate 0.3 --max_bin 16 --feature_fraction 0.15
python src/scripts/lightgbm_python/score.py --data ./data/synthetic/inference/ --model ./data/models/synthetic-1200/ --output ./data/outputs/predictions/ --nthread 4
