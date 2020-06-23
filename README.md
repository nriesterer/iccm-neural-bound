ICCM Neural Bounds
==================

Companion repository for the 2019 article "Predictive Modeling of Individual Human Cognition: Upper Bounds and a New Perspective on Performance" published for the 17th Annual Meeting of the International Conference on Cognitive Modelling.

## Overview

- `baseline/`: Syllogistic baseline models
    - `baseline/Khemlani2012`: Models from Khemlani & Johnson-Laird's 2012 meta analysis "Theories of the syllogism: A meta-analysis"
    - `baseline/MFA-Model`: Most-frequent answer model
    - `baseline/Uniform-Model`: Uniform random model
- `data/`: Evaluation data
    - `data/Ragni-test.csv`: Test data split (N=39) of the Ragni2016 dataset
    - `data/Ragni-train.csv`: Train data split (N=100) of the Ragni2016 dataset
    - `data/Ragni2016.csv`: Ragni 2016 dataset
    - `data/split.py`: Script to split the Ragni 2016 data into training and test data sets
- `networks/`: Network model implementations
    - `networks/autoencoder`: Autoencoder model implementation
    - `networks/mlp`: Adaptive multi-layer perceptron model implementation
    - `networks/rnn`: Recurrent neural network model implementation

## Dependencies

- Python 3
    - [CCOBRA 1.1.0](https://github.com/CognitiveComputationLab/ccobra)
    - [pandas](https://pandas.pydata.org)
    - [numpy](https://www.numpy.org)
    - [seaborn](https://seaborn.pydata.org)

## Quickstart

### CCOBRA Analysis (Figure 1)

Navigate into the `analysis/` folder and run CCOBRA on the benchmark files:

```
$> cd /path/to/analysis/
$> ccobra bench.json
```

### Network Training Analysis (Figure 2)

Navigate into the respective network subfolder (`analysis/networks/autoencoder`, `analysis/networks/mlp`, `analysis/networks/rnn`) and execute the `train_eval.py` (collects the loss data) and `train_eval_plot.py` scripts:

```
$> cd /path/to/networks/folder
$> python train_eval.py
$> python train_eval_plot.py
```

## Reference

Riesterer, N., Brand, D., & Ragni, M. (2019). Predictive Modeling of Individual Human Cognition: Upper Bounds and a New Perspective on Performance. In Stewart T. (Ed.), Proceedings of the 17th International Conference on Cognitive Modeling.
