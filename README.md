<p align="center" style="margin: 0; padding: 0;">
  <img src="https://github.com/gregory-kyro/CardioGenAI/assets/98780179/e04ca9a0-0340-440f-a87c-a417e7136fb1" alt="cardiogenai_logo">
</p>

[![](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/gregory-kyro/CardioGenAI/blob/main/LICENSE)

## Summary
Drug-induced cardiotoxicity is a major health concern which can lead to serious adverse effects including life-threatening cardiac arrhythmias via the blockade of cardiac ion channels such as hERG, NaV1.5 and CaV1.2. It is therefore of tremendous interest to develop advanced methods to identify cardiotoxic compounds in early stages of drug development, as well as to optimize commercially available drugs for reduced cardiac ion channel activity. In this work, we present CardioGenAI, a machine learning-based framework for re-engineering both developmental and marketed drugs for reduced cardiotoxicity while preserving their pharmacological activity. The framework incorporates novel state-of-the-art discriminative models for predicting hERG, NaV1.5 and CaV1.2 channel activity, which can also serve independently as effective components of a virtual screening pipeline. We applied the complete framework to pimozide, an FDA-approved antipsychotic agent that demonstrates high affinity to the hERG channel, and generated 100 refined candidates. Remarkably, among the candidates is fluspirilene, a compound which is of the same class of drugs (diphenylmethanes) as pimozide and therefore has similar pharmacological activity, yet exhibits over 700-fold weaker binding to hERG. We have made all of our software open-source to facilitate implementation.

![cgaicf](https://github.com/gregory-kyro/CardioGenAI/assets/98780179/960a9e40-1c98-45c8-a770-19bb2366f647)

## Technical Overview of the Framework
The CardioGenAI framework combines generative and discriminative ML models to re-engineer cardiotoxic compounds for reduced cardiac ion channel activity while preserving their pharmacological action. An autoregressive transformer is trained on a dataset that we previously curated which contains approximately 5 million unique and valid SMILES strings derived from ChEMBL 33, GuacaMol v1, MOSES, and BindingDB datasets. The model is trained autoregressively, receiving a sequence of SMILES tokens as context as well as the corresponding molecular scaffold and ADMET properties, and iteratively predicting each subsequent token in the sequence. Once trained, this model is able to generate valid molecules conditioned on a specified molecular scaffold along with a set of ADMET properties. For an input cardiotoxic compound, the generation is conditioned on the scaffold and ADMET properties of this compound. Each generated compound is subject to filtering based on activity against hERG, NaV1.5 and CaV1.2 channels. Depending on the desired activity against each channel, the framework employs either classification models to include predicted non-blockers or regression models to include compounds within a specified range of predicted pIC50 values. Both the classification and regression models utilize the same architecture, and are trained using three feature representations of each molecule: a feature vector that is extracted from a bidirectional transformer trained on SMILES strings, a molecular fingerprint, and a graph. For each molecule in the filtered generated ensemble and the input cardiotoxic molecule, a feature vector is constructed from the 209 chemical descriptors available through the RDKit Descriptors module. The redundant descriptors are then removed according to pairwise mutual information calculated for every possible pair of descriptors. Cosine similarity is then calculated between the processed descriptor vector of the input molecule and the descriptor vectors of every generated molecule to identify the molecules most chemically similar to the input molecule, but with desired activity against each of the cardiac ion channels.

## Installation and Setup
Follow these instructions to install and set up CardioGenAI on your local machine:

### Cloning the Repository
Clone the CardioGenAI repository to your local environment using the following command:

```
git clone https://github.com/gregory-kyro/CardioGenAI.git
```

After cloning, navigate to the CardioGenAI project directory:

```
cd CardioGenAI
```

### Setting Up the Conda Environment
Create a Conda environment using the `environment.yml` file provided in the repository which contains all of the necessary dependencies:

```
conda env create -f environment.yml
```

Activate the newly created environment:

```
conda activate cardiogenai_env
```

### Downloading Necessary Files
Some essential files are not hosted directly in the GitHub repository due to their sizes. Please download the following files from the provided Google Drive links:

- [Autoregressive_Transformer_parameters.pt](https://drive.google.com/file/d/1oj2OkjRNX3rYN9xv0GKkANjWKf1ebLLN/view?usp=sharing)
- [prepared_transformer_data.csv](https://drive.google.com/file/d/1l2Osk7zFj4rTyrjAi7EJ1GMrsYMbcRHI/view?usp=drive_link)
- [raw_transformer_data.csv](https://drive.google.com/file/d/1pVOFnNT2sfLRaLoHnF-qDCs6G_worX0e/view?usp=drive_link)
- [train_hERG.h5](https://drive.google.com/file/d/1xfNwpVIhqWyFW_3z3sUyuy-45i248J-0/view?usp=drive_link)

After downloading, place these files in the specified directories within the CardioGenAI project:

- `Autoregressive_Transformer_parameters.pt` → `model_parameters/transformer_model_parameters`
- `prepared_transformer_data.csv` → `data/prepared_transformer_datasets`
- `raw_transformer_data.csv` → `data/raw_transformer_datasets`
- `train_hERG.h5` → `data/prepared_cardiac_datasets/`

# Running the Software
Running the complete CardioGenAI framework, performing inference with the discriminative models, and reproducing the figures in the manuscript can easily be achieved with the [Jupyter notebook](https://github.com/gregory-kyro/CardioGenAI/blob/main/_run.ipynb) provided with this repository. Simply navigate to the CardioGenAI project directory, open the `_run.ipynb` notebook, and select the `cardiogenai_env` environment as the kernel. Usage instructions are below.

## Running the CardioGenAI Framework
To optimize a cardiotoxic compound with CardioGenAI, utilize the `optimize_cardiotoxic_drug` function from the `Optimization_Framework` module:

```
from src.Optimization_Framework import optimize_cardiotoxic_drug

optimize_cardiotoxic_drug(input_smiles,
                          herg_activity,
                          nav_activity,
                          cav_activity,
                          n_generations,
                          device)
```

- `input_smiles (str)`: The input SMILES string of the compound that you seek to optimize for reduced cardiotoxicity.
- `herg_activity (tuple or str)`: hERG activity for which to filter. If the entry is a string, it must be either 'blockers' or 'non-blockers'. If it is a tuple, it must indicate a range of activity values.
- `nav_activity (tuple or str)`: NaV1.5 activity for which to filter. If the entry is a string, it must be either 'blockers' or 'non-blockers'. If it is a tuple, it must indicate a range of activity values.
- `cav_activity (tuple or str)`: CaV1.2 activity for which to filter. If the entry is a string, it must be either 'blockers' or 'non-blockers'. If it is a tuple, it must indicate a range of activity values.
- `n_generations (int)`: The number of optimized drug candidates to generate. Default is 100.
- `device (str)`: The device to use for the optimization. Must be either 'gpu' or 'cpu'. Default is 'gpu'.


## Performing Inference with the Discriminative Models

To predict activity against the hERG, NaV1.5 and CaV1.2 channels, utilize the `predict_cardiac_ion_channel_activity` function from the `Discriminator` module:

```
from src.Discriminator import predict_cardiac_ion_channel_activity

predict_cardiac_ion_channel_activity(input_data,
                                     prediction_type,
                                     predict_hERG,
                                     predict_Nav,
                                     predict_Cav,
                                     device)
```

- `input_data (str or list)`: The input data for which the discriminative models will process. If the entry is a string, it must be either a SMILES string or a path to a prepared h5 file. If it is a list, it must be a list of SMILES strings.
- `prediction_type (str)`: Either 'regression' or 'classification'. Default is 'regression'.
- `predict_hERG (bool)`: Whether to predict hERG activity. Default is True.
- `predict_Nav (bool)`: Whether to predict NaV1.5 activity. Default is False.
- `predict_Cav (bool)`: Whether to predict CaV1.2 activity. Default is False.
- `device (str)`: The device to use for the inference computations. Must be either 'gpu' or 'cpu'. Default is 'gpu'.


## Reproducing the Figures in the Manuscript

To reproduce the results presented in the manuscript, utilize the `get_figures` function from the `Figures` module:

```
from src.Figures import get_figures

get_figures()
```
