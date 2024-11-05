<div align="center">

# SynTPack (Synthetic Table Package)

<div align="left">

## What is this repo about?

This repository is designed to make it easy for users to generate synthetic tables with minimal setup. This repo integrates popular methods like ADASYN and SMOTE for resampling, as well as advanced generative models such as GAN, Autoencoder, and Transformer-based models for synthesizing realistic, high-dimensional tabular data. Each method is implemented to help users quickly create balanced datasets or generate synthetic samples based on an existing table, without needing to write extensive code.

With this toolkit, you can:

- Choose from multiple synthesis methods based on your dataset needs—whether you're resampling with traditional techniques or using generative models.
- Easily customize parameters such as target distributions, sample sizes, and more for different synthesis approaches.
- Visualize synthetic data distributions and correlations with built-in plotting functions to assess the output.

This repo is meant to streamline the process of synthetic data generation, helping users focus on their analysis without the overhead of extensive coding.

## Before using this repo...

First, clone this repo to your local machine. Then, create a new conda environment with its dependencies by executing the following syntax in terminal or console.

> Note: It is strongly urged to use this package on NVIDIA GPU-powered machine since most of the methods used in this repo requires PyTorch/Tensorflow.

```
conda env create --file requirements.yml
```

After a new conda environment created, activate the environment by

```
conda activate SynTPack
```

Transformer-based synthetic table generator is still not available in conda package manager. While in the activated `SynTPack` environment, install REaLTabFormer package with `pip`.

```
pip install realtabformer
```

## How to use this repo?

Using this repo requires no more than 10 lines, assuming you have already your own dataframe as a reference.

After cloning this repo on your local machine, installing the required dependencies, and ensuring this cloned repo as your current working directory, you can generate synthetic table with following example.

### For Generating Synthetic Table

```
from syntpack.syntpack import SynTable

# Your previous code
# ...
# Assuming you have your own 'df' as your dataframe

# The available methods are 'adasyn', 'smote', 'ctgan', 'tvae', 'rtf'

# Let's say, we use 'CTGAN' to generate synthetic table
# Assuming that the synthetic table is generated based on 'target' column

# Let's say in your 'target' column, there are three class,
# 'Circle', 'Square', 'Triangle', and you want the classes distribution
# to be 3, 3, 4 respectively. You can create an appropriate dictionary like
# {'Circle': 0.3, 'Square': 0.3, 'Triangle': 0.4}.
# Ensure that the sum of dict values no more than 1.

# Now, comes the real code
tabgen = SynTable(df, target_col='target', method='ctgan', target_conditions={'Circle': 0.3, 'Square': 0.3, 'Triangle': 0.4})

# You can generate synthetic table of a certain number of rows. Let's say 1000 rows.
df_synthetic = tabgen.synthesize(num_samples=1000)

# You can specify the number of epochs, batch_size, logging_stepsor
# or turning on log_frequency as you see fit.
# Here is the full synthesize method.
# tabgen.synthesize(num_samples=1000, epochs=300, batch_size=30, log_frequency=False, logging_steps=20)
```

### Using SynTPack's Plotting libraries

```
from syntpack.syntpack import SynPlot

# Your previous code
# ...
# Assuming you want to plot the features of your 'df' dataframe

synfigs = SynPlot(df)

# To plot the distribution of categorical features, you can run
synfigs.categorical_dist()

# To plot the distribution of numerical features, you can run
synfigs.numerical_dist()

# To plot the Cramer's correlation of categorical features, you can run
synfigs.cramer_corr()

# To plot the Pearson correlation of numerical features, you can run
synfigs.pearson_corr()
```

## How about the performance?

Coming soon!

## Don't Skip This!

There are many things that still should be fixed within this repo. So, don't hesitates to report an issue if you encounter any!
