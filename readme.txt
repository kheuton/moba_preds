This repository contains all the code necessary to train, evaluate, and use a model that will predict win rates
based on DotA team compositions. The contents of this repo are listed below in order of interestingness:


- evaluate_team.py:
    This is the main tool. Call this with "python evaluate_team.py", and respond to the prompts to enter the lineups.
    If you are not familiar with DotA heros, some will be suggested. The tool will output the expected win rate for the
    Radiant team (the team that starts on the left of the map), as well as substitutions the Radiant could consider
    making. This file requires a python environment containing tensorflow 2.4

- "Test model.ipynb":
    This Jupyter notebook fits the final model using hyperparameters selected in our grid search.
    If you do not have an environment that can run this notebook, it is available online at
    https://colab.research.google.com/drive/1dUiqNbGnVjAtaJpTmTrTrQ6IkzFp7Ph3
    However, the online version requires a Kaggle account to download the dataset.
    After running this notebook, you can use Tensorboard to inspect the training and validation dataset loss,
    as well as explore PCA and t-SNE projections of the hero embedding.

- "model.py":
    This file contains the tensorflow code defining our predictive model. Each team is run through a shared embedding
    layer, and then a fully-connected neural classifier predicts win probability based on each lineup's embedding and
    the average historical winrate of the heroes on the team.

- "Launch Jobs.ipynb"
    This notebook was used to run our hyperparamter grid search on the Tufts HPC.

- "test_run.py", "run_experiment.slurm"
    These files controlled the individual runs of the grid search on the HPC

- data
    This directory contains 400k+ DotA games from the https://www.kaggle.com/competitions/dota-2-prediction/data
    Kaggle competition

- logs
    This directory contains training logs and the embedding layer of our final model

- results
    This directory contains historical winrate data, as well as the final model weights used in evaluate_team.py
