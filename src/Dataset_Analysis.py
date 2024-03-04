# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from tqdm import tqdm

from src.utils import get_admet_properties


# Define a function to analyze the cardiac ion channel dataset
def analyze_cardiac_ion_channel_dataset(
    dataset="data/raw_cardiac_datasets/raw_train_val_hERG.csv",
    save_dir="results/dataset_analysis_results/",
):
    """
    Analyzes a cardiac ion channel dataset.

    Args:
        dataset (str): The path to the dataset file.
        save_dir (str, optional): The directory to save the analysis results. Defaults to 'dataset_analysis_results/'.

    Returns:
        None
    """

    os.makedirs(save_dir, exist_ok=True)

    # Load the dataset and calculate ADMET properties
    df = pd.read_csv(dataset)
    admet_data = [
        get_admet_properties(smiles)
        for smiles in tqdm(df["SMILES"], desc="Calculating ADMET properties")
    ]

    # Define the features and target variable
    X = pd.DataFrame(
        admet_data,
        columns=[
            "MW",
            "HBA",
            "HBD",
            "nRot",
            "nRing",
            "nHet",
            "fChar",
            "TPSA",
            "LogP",
            "StereoCenters",
        ],
    )
    y = df["pIC50"]

    # Calculate Pearson correlation coefficients
    correlations = X.corrwith(y)

    corr_df = pd.DataFrame(correlations, columns=["Correlation"])
    corr_df["Feature"] = corr_df.index
    corr_df = corr_df.reset_index(drop=True)
    corr_df = corr_df[["Feature", "Correlation"]]

    # Save the correlation coefficients to a CSV file
    corr_df.to_csv(os.path.join(save_dir, "correlations.csv"))
    print("Pearson Correlation Coefficients:\n", correlations)

    # Train a random forest model on the ADMET properties to predict pIC50
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Plot the feature importances
    importance = rf.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(
        range(X.shape[1]), importance[sorted_idx], align="center", color="steelblue"
    )
    plt.yticks(range(X.shape[1]), X.columns[sorted_idx], fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlabel("Feature Importance", fontsize=16)
    plt.title(
        "Random Forest Feature Importance for pIC50 Prediction",
        fontsize=18,
        weight="bold",
    )
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_importance.png"))
    plt.show()

    # Plot the partial dependence plots
    _, axs = plt.subplots(1, 2, figsize=(15, 6))
    for i, ax in enumerate(axs):
        pd_results = partial_dependence(rf, X, features=[sorted_idx[i]])
        XX, YY = np.meshgrid(pd_results["grid_values"][0], pd_results["average"][0])
        ax.plot(XX[0], YY, color="darkgreen")
        ax.set_title(
            f"Partial Dependence for {X.columns[sorted_idx[i]]}",
            fontsize=20,
            weight="bold",
        )
        ax.set_xlabel(X.columns[sorted_idx[i]], fontsize=16)
        ax.set_ylabel("Partial Dependence", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "partial_dependence.png"))
    plt.show()


# Define a function to analyze the top and bottom pIC50 compounds
def analyze_top_and_bottom_pIC50_compounds(
    dataset="data/raw_cardiac_datasets/raw_train_val_hERG.csv",
    save_dir="results/dataset_analysis_results/",
):
    """
    Analyzes the top and bottom compounds based on pIC50 values in a given dataset.

    Args:
        dataset (str): The path to the dataset file.
        save_dir (str, optional): The directory to save the analysis results. Defaults to '/dataset_analysis_results/'.

    Returns:
        PIL.Image.Image: The image containing the visual representation of the top and bottom compounds.

    """

    os.makedirs(save_dir, exist_ok=True)

    # Load the dataset and select the top and bottom compounds
    compounds_df = pd.read_csv(dataset)
    top_compounds = compounds_df.nlargest(5, "pIC50")
    bottom_compounds = compounds_df.nsmallest(5, "pIC50")

    selected_compounds = pd.concat([top_compounds, bottom_compounds])
    selected_compounds["Molecule"] = selected_compounds["SMILES"].apply(
        Chem.MolFromSmiles
    )
    top_mols = [Chem.MolFromSmiles(smiles) for smiles in top_compounds["SMILES"]]
    bottom_mols = [Chem.MolFromSmiles(smiles) for smiles in bottom_compounds["SMILES"]]

    # Draw the top and bottom compounds and return the image
    img = Draw.MolsToGridImage(
        top_mols + bottom_mols,
        molsPerRow=5,
        subImgSize=(400, 400),
        legends=[f"{p:.3f}" for p in selected_compounds["pIC50"]],
    )

    return img
