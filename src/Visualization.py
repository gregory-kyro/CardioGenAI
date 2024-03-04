# Import necessary libraries
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib_venn import venn3
import numpy as np
import os
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
from scipy.stats import gaussian_kde
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import warnings

from src.Discriminator import screen_FDA_compounds
from src.Transformer import Transformer_Dataset, Transformer_Model
from src.utils import get_admet_properties


# Define a function to project the transformer dataset, generated molecules, and target molecule into an ADMET-based PCA space
def plot_pca_space(
    transformer_dataset="data/prepared_transformer_datasets/prepared_transformer_data.csv",
    generated_dataset="results/optimization_results/pimozide_optimized_drugs.csv",
    target_smiles="O=c1[nH]c2ccccc2n1C1CCN(CCCC(c2ccc(F)cc2)c2ccc(F)cc2)CC1",
    save_dir="results/visualization_results/",
):
    """
    Plots the 2-dimensional PCA plot of ADMET-based chemical space and the variance explained by principal components (PCs) of ADMET-based space.

    Args:
        transformer_dataset (str): Path to the transformer dataset CSV file. Default is 'prepared_transformer_datasets/prepared_transformer_data.csv'.
        generated_dataset (str): Path to the generated dataset CSV file. Default is None.
        target_smiles (str): SMILES representation of the target molecule. Default is None.
        save_dir (str): Directory to save the visualization results. Default is 'visualization_results/'.

    Returns:
        None
    """

    print(
        "Performing PCA on the ADMET properties of the datasets... (This will take about 15 seconds)"
    )

    os.makedirs(save_dir, exist_ok=True)

    # Load the datasets
    transformer_data = pd.read_csv(transformer_dataset)
    generated_data = pd.read_csv(generated_dataset)
    target_data = pd.DataFrame(
        [get_admet_properties(target_smiles)], columns=transformer_data.columns[1:-1]
    )
    combined_data = pd.concat(
        [transformer_data.iloc[:, 1:-1], generated_data.iloc[:, 1:-1], target_data]
    )

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data)

    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)

    num_transformer_datapoints, num_generated_datapoints = len(transformer_data), len(
        generated_data
    )

    # Define the variance explained by the first two principal components
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100

    # Plot the PCA biplot
    plt.figure(figsize=(10, 8))
    plt.scatter(
        pca_result[:num_transformer_datapoints, 0],
        pca_result[:num_transformer_datapoints, 1],
        alpha=1,
        color="lightcoral",
        s=1,
    )
    plt.scatter(
        pca_result[
            num_transformer_datapoints : num_transformer_datapoints
            + num_generated_datapoints,
            0,
        ],
        pca_result[
            num_transformer_datapoints : num_transformer_datapoints
            + num_generated_datapoints,
            1,
        ],
        alpha=1,
        color="darkmagenta",
        s=1,
    )
    plt.scatter(pca_result[-1, 0], pca_result[-1, 1], alpha=1, color="yellow", s=20)
    plt.title(
        "2-Dimensional PCA Plot of ADMET-based Chemical Space",
        fontsize=20,
        weight="bold",
    )
    plt.xlabel(f"PC1 ({var_pc1:.2f}% Variance Explained)", fontsize=16)
    plt.ylabel(f"PC2 ({var_pc2:.2f}% Variance Explained)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Create a custom legend
    legend_transformer = mlines.Line2D(
        [],
        [],
        color="lightcoral",
        marker="o",
        linestyle="None",
        markersize=10,
        label="Transformer pretraining set",
    )
    legend_generated = mlines.Line2D(
        [],
        [],
        color="darkmagenta",
        marker="o",
        linestyle="None",
        markersize=10,
        label="Generated molecules",
    )
    legend_input = mlines.Line2D(
        [],
        [],
        color="yellow",
        marker="o",
        linestyle="None",
        markersize=10,
        label="Input molecule",
    )

    plt.legend(
        handles=[legend_transformer, legend_generated, legend_input],
        loc="lower right",
        fontsize=14,
    )
    plt.savefig(os.path.join(save_dir, "2d_pca_admet_space.png"))
    plt.show()

    # Plot the variance explained by the principal components
    plt.figure(figsize=(8, 6))
    plt.plot(
        np.cumsum(pca.explained_variance_ratio_) * 100,
        marker="o",
        linestyle="-",
        color="royalblue",
    )
    plt.title(
        "Variance Explained by PCs of ADMET-based Space",
        fontsize=20,
        weight="bold",
        pad=20,
    )
    plt.xlabel("Number of Components", fontsize=16)
    plt.ylabel("Cumulative Explained Variance (%)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, 105)
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "variance_explained_pca_admet_space.png"))
    plt.show()


# Define a function to plot the pIC50 values of the generated compounds
def plot_pIC50s_of_generated_compounds(
    generated_smiles="results/optimization_results/pimozide_optimized_drugs.csv",
    save_dir="results/visualization_results/",
):
    """
    Plots the pIC50 values of generated compounds.

    Parameters:
    - generated_smiles (str): Path to the file containing the generated SMILES and pIC50 values.
                              Default: 'optimization_results/pimozide_optimized_drugs.csv'
    """

    pic50_values = []

    # Read the pIC50 values from the input file of generated SMILES
    with open(generated_smiles, "r") as file:
        lines = file.readlines()
        for ind, line in enumerate(lines):
            if ind == 0 or ind == 1:
                continue
            fields = line.strip().split(",")
            if len(fields) > 1:
                pic50 = float(fields[-1])
                pic50_values.append(pic50)

    # Obtain the minimum, maximum, and mean pIC50 values
    min_value = np.min(pic50_values)
    max_value = np.max(pic50_values)
    mean_value = np.mean(pic50_values)

    # Plot the histogram of pIC50 values
    plt.figure(figsize=(10, 6))
    plt.hist(
        pic50_values,
        bins=100,
        density=True,
        alpha=0.6,
        color="skyblue",
        edgecolor="black",
    )
    plt.title(
        "pIC50 Values of Generated Molecules", fontsize=22, fontweight="bold", pad=15
    )
    plt.xlabel("pIC50", fontsize=16)
    plt.ylabel("Count", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    legend_text = (
        f"Minimum: {min_value:.2f}\nMaximum: {max_value:.2f}\nMean: {mean_value:.2f}"
    )
    plt.text(
        0.13,
        0.85,
        legend_text,
        ha="left",
        va="top",
        transform=plt.gca().transAxes,
        fontsize=18,
    )
    plt.grid(False)
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(save_dir, "pIC50_histogram.png"))
    plt.show()


# Define a function to plot the attention weights of one of the transformer models
def plot_attention_weights(
    model_type="Autoregressive",
    model_params=None,
    training_data=None,
    smiles=None,
    save_dir=None,
    device="gpu",
):
    """
    Plot the attention weights for a given model.

    Args:
        model_type (str): The type of model. Must be either 'Autoregressive' or 'Bidirectional'.
        model_params (str): The path to the model parameters file.
        training_data (str): The path to the training data.
        smiles (str): The SMILES string for which to compute the attention weights.
        save_dir (str): The directory to save the plot.
        device (str): The device to use for computation. Must be either 'cpu' or 'gpu'.

    Returns:
        None
    """

    assert model_type in [
        "Autoregressive",
        "Bidirectional",
    ], "Model type must be either 'Autoregressive' or 'Bidirectional'"
    assert device in ["cpu", "gpu"], "Device must be either 'cpu' or 'gpu'"

    # Supress warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(
        "cuda:0" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
    )

    # Load the dataset and model
    print("Loading model... (This will take about 1 minute)")
    dataset = Transformer_Dataset(mode=model_type, data_path=training_data)
    model = (
        Transformer_Model(
            mode=model_type,
            vocab_size=len(dataset.vocab),
            block_size=dataset.block_size,
            admet_dim=10,
            num_scaffolds=dataset.num_scaffolds,
        ).to(device)
        if model_type == "Autoregressive"
        else Transformer_Model(
            mode=model_type,
            vocab_size=len(dataset.vocab),
            block_size=dataset.block_size,
        ).to(device)
    )
    checkpoint = torch.load(model_params, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    # Get the token indices for the input SMILES string
    tokens = [char for char in smiles]
    token_idx = [dataset.stoi[s] for s in dataset.smiles_regex.findall(smiles)]
    token_idx_tensor = torch.tensor(token_idx, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        if model_type == "Autoregressive":

            # Create dummy input for ADMET properties and scaffold index
            dummy_admet_props = torch.zeros(1, 10).to(device)
            dummy_scaffold_idx = torch.tensor([0], dtype=torch.long).to(device)

            # Perform a forward pass to obtain the attention weights
            _, attention_weights = model(
                token_idx_tensor,
                dummy_admet_props,
                dummy_scaffold_idx,
                return_attention_weights=True,
            )

        elif model_type == "Bidirectional":

            # Perform a forward pass to obtain the attention weights
            _, attention_weights = model(
                token_idx_tensor, return_attention_weights=True
            )

    # Format the attention weights
    attention_weights_array = np.array(
        [tensor.cpu().numpy() for tensor in attention_weights]
    )
    attention_weights_array = np.squeeze(attention_weights_array, axis=1)
    num_heads = attention_weights_array.shape[1]

    # Plot the attention weights for each head
    fig, axes = plt.subplots(4, 2, figsize=(10, 20))
    color = "Greens" if model_type == "Autoregressive" else "Purples"
    for i in range(num_heads):
        ax = axes[i // 2, i % 2]
        sns.heatmap(attention_weights_array[0, i], ax=ax, cmap=color, cbar=False)
        ax.set_title(f"Head {i+1}", fontsize=16)
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens)
        ax.set_yticklabels(tokens)
        ax.set_xlabel("Key Vector", fontsize=14)
        ax.set_ylabel("Query Vector", fontsize=14)
        ax.plot([0, len(tokens) - 1], [0, len(tokens) - 1], color="gray", linewidth=1)
    cbar_ax = fig.add_axes([1, 0.15, 0.02, 0.7])
    plt.colorbar(axes[0, 0].collections[0], cax=cbar_ax)
    title = f"Attention Weights for {model_type} Transformer"
    fig.suptitle(title, fontsize=20, y=1, weight="bold")
    plt.tight_layout()

    # Save the plot
    plt.savefig(
        os.path.join(save_dir, f"{model_type}_transformer_attention_weights.png")
    )
    plt.show()


# Define a function to plot heatmaps of attention weights for both transformer models
def plot_attention_weights_for_both_models(
    parameters_directory="model_parameters/transformer_model_parameters",
    training_data="data/prepared_transformer_datasets/prepared_transformer_data.csv",
    smiles="CCC(=O)CCNC(C)C(=O)c1ccncc1C",
    save_dir="results/visualization_results",
    device="gpu",
):
    """
    Plots attention weights for both Bidirectional and Autoregressive models.

    Args:
        parameters_directory (str): Directory path where the model parameters are stored.
        training_data (str): File path of the training data.
        smiles (str): SMILES representation of the molecule to visualize attention weights for.
        save_dir (str): Directory path to save the visualization results.
        device (str): Device to use for computation (e.g., 'cpu', 'gpu').
    """

    for file in os.listdir(parameters_directory):
        if "Autoregressive" in file:
            autorergessive_params = os.path.join(parameters_directory, file)

        elif "Bidirectional" in file:
            bidirectional_params = os.path.join(parameters_directory, file)

    # Plot the attention weights for the autoregressive model
    plot_attention_weights(
        model_type="Bidirectional",
        model_params=bidirectional_params,
        training_data=training_data,
        smiles=smiles,
        save_dir=save_dir,
        device=device,
    )

    # Plot the attention weights for the bidirectional model
    plot_attention_weights(
        model_type="Autoregressive",
        model_params=autorergessive_params,
        training_data=training_data,
        smiles=smiles,
        save_dir=save_dir,
        device=device,
    )


# Define a function to plot learning curves for the discriminative and transformer models
def plot_learning_curves(
    discriminative_params=[
        "model_parameters/discriminative_model_parameters/hERG_Classification_parameters.pt",
        "model_parameters/discriminative_model_parameters/Nav_Classification_parameters.pt",
        "model_parameters/discriminative_model_parameters/Cav_Classification_parameters.pt",
        "model_parameters/discriminative_model_parameters/hERG_Regression_parameters.pt",
        "model_parameters/discriminative_model_parameters/Nav_Regression_parameters.pt",
        "model_parameters/discriminative_model_parameters/Cav_Regression_parameters.pt",
    ],
    transformer_params=[
        "model_parameters/transformer_model_parameters/Bidirectional_Transformer_parameters.pt",
        "model_parameters/transformer_model_parameters/Autoregressive_Transformer_parameters.pt",
    ],
    save_dir="results/visualization_results",
):
    """
    Plots learning curves for classification and regression models, as well as autoregressive and bidirectional transformer models.

    Args:
        discriminative_params (list): List of file paths for discriminative models.
        transformer_params (list): List of file paths for transformer models.
        save_dir (str): Directory to save the generated plot. Defaults to 'visualization_results'.

    Returns:
        None
    """

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    _, axs = plt.subplots(2, 2, figsize=(20, 12))

    # Define the color sets for the discriminative and transformer models
    colorset_1, colorset_2 = ["#030764", "#054907", "#FF796C"], ["#030764", "#054907"]

    titles = [
        "Classification Learning Curves",
        "Regression Learning Curves",
        "Autoregressive Transformer Learning Curve",
        "Bidirectional Transformer Learning Curve",
    ]

    # Define the labels for the discriminative models
    labels_map = {"hERG": "hERG", "Nav": "Nav1.5", "Cav": "Cav1.2"}

    for file_path in discriminative_params:
        if os.path.exists(file_path):

            # Load the discriminative model training losses
            training_losses = torch.load(file_path)["training_losses"]

            label_key = file_path.split("/")[-1].split("_")[0]
            label = labels_map.get(label_key, label_key)

            color_index = list(labels_map.keys()).index(label_key) % len(colorset_1)

            if "Classification" in file_path:

                # Plot the classification learning curves
                axs[0, 0].plot(
                    training_losses,
                    label=label,
                    color=colorset_1[color_index],
                    linewidth=2,
                )
            else:

                # Plot the regression learning curves
                axs[0, 1].plot(
                    training_losses,
                    label=label,
                    color=colorset_1[color_index],
                    linewidth=2,
                )

    for file_path in transformer_params:
        if os.path.exists(file_path):

            # Load training and validation losses for the transformer models
            model_state_dict = torch.load(file_path)
            training_losses = model_state_dict["training_losses"]
            validation_losses = model_state_dict["validation_losses"]

            epochs = range(len(training_losses))
            model_type = (
                "Autoregressive" if "Autoregressive" in file_path else "Bidirectional"
            )

            # Plot the transformer learning curves
            idx = 0 if "Autoregressive" in file_path else 1
            axs[1, idx].plot(
                epochs,
                training_losses,
                label="Training Loss",
                color=colorset_2[0],
                linewidth=2,
            )
            axs[1, idx].plot(
                epochs,
                validation_losses,
                label="Validation Loss",
                color=colorset_2[1],
                linewidth=2,
            )
            axs[1, idx].set_title(
                f"{model_type} Transformer Learning Curve",
                fontsize=20,
                fontweight="bold",
            )
    for i, ax in enumerate(axs.flat):
        ax.set_title(titles[i], fontsize=20, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=16)
        ax.set_ylabel("Loss", fontsize=16)
        ax.legend(fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "learning_curves.png"))
    plt.show()


# Define a function to plot the distribution of generated compounds conditioned on ADMET properties
def plot_conditional_generation_distribution(
    descriptors_dict, ax, files, descriptor_key
):
    """
    Plots the conditional generation distribution based on the given descriptors.

    Args:
        descriptors_dict (dict): A dictionary containing information about the descriptors.
        ax (matplotlib.axes.Axes): The axes object to plot on.
        files (list): A list of file paths containing SMILES strings.
        descriptor_key (str): The key of the descriptor to plot.

    Returns:
        None
    """

    colors = ["#4B0082", "#8B0000"]

    # Retrieve the descriptor information for each ADMET property
    descriptor_info = descriptors_dict[descriptor_key]

    for i, file_path in enumerate(files):
        with open(file_path, "r") as f:

            # Load the SMILES strings from the input file
            smiles_list = f.read().splitlines()

        # Calculate the ADMET properties for each SMILES string
        descriptor_values = [
            get_admet_properties(smiles)[descriptor_info["index"]]
            for smiles in smiles_list
            if Chem.MolFromSmiles(smiles)
        ]

        # Define the condition value which is included in the file name
        condition_value = int("".join(filter(str.isdigit, os.path.basename(file_path))))

        color = colors[i % len(colors)]

        # Plot the distribution of the ADMET properties
        num_bins = 20
        bin_width = (max(descriptor_values) - min(descriptor_values)) / num_bins
        total_count = len(descriptor_values)
        kde = gaussian_kde(descriptor_values, bw_method=descriptor_info["bw"])

        x_range = np.linspace(min(descriptor_values), max(descriptor_values), 1000)
        y_range = kde(x_range) * total_count * bin_width

        ax.fill_between(x_range, y_range, color=color, alpha=0.6)
        ax.plot(x_range, y_range, color=color, linewidth=3, label=str(condition_value))

        # Obtain the mode value and plot it on the distribution
        mode_index = np.argmax(y_range)
        mode_value = x_range[mode_index]
        text_y = max(y_range)
        ax.text(
            mode_value,
            text_y + 3,
            f"{np.round(mode_value,0):.1f}",
            horizontalalignment="center",
            color=color,
        )

        ax.set_ylim(0, max(y_range) + max(y_range) * 0.3)
        ax.set_xlim(min(descriptor_values), max(descriptor_values))

    ax.set_title(descriptor_info["name"], fontsize=20, weight="bold")
    ax.set_xlabel(descriptor_info["name"], fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.legend(title="Condition", fontsize=16, title_fontsize=16)


# Define a function to plot the distribution of generated compounds conditioned on multiple ADMET properties
def plot_multiconditional_generation_distribution(directory_path, save_dir=None):
    """
    Plots a multiconditional generation distribution based on the given directory path.

    Args:
        directory_path (str): The path to the directory containing the input files.
        save_dir (str, optional): The directory where the plot image will be saved. Defaults to None.

    Returns:
        None
    """

    # Supress RDKit warnings
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    colors = ["#4B0082", "#F0E68C", "#FF69B4", "#CD5C5C"]

    all_data = []

    # Obtain the files that compound conditioned on both MW and TPSA values
    smiles_files = [
        f for f in os.listdir(directory_path) if (f.startswith("mw") and "tpsa" in f)
    ]

    for _, file_name in enumerate(smiles_files):
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, "r") as file:

            # Load the SMILES strings from the input file
            smiles_list = file.read().splitlines()

        mw_values, tpsa_values = [], []
        condition_parts = file_name.replace(".txt", "").split("_")
        formatted_condition = "MW: {}; TPSA: {}".format(
            condition_parts[0][2:], condition_parts[1][4:]
        )

        for smiles in smiles_list:

            # Create an RDKit molecule object from the SMILES string
            mol = Chem.MolFromSmiles(smiles)
            if mol:

                # Calculate the molecular weight and TPSA
                mw_values.append(Descriptors.MolWt(mol))
                tpsa_values.append(Descriptors.TPSA(mol))

        # Calculate the mean and standard deviation of the MW and TPSA values
        mw_mean, mw_std = np.mean(mw_values), np.std(mw_values)
        tpsa_mean, tpsa_std = np.mean(tpsa_values), np.std(tpsa_values)

        # Filter the data based on the mean and standard deviation values of MW and TPSA
        filtered_data = [
            {"Molecular Weight": mw, "TPSA": tpsa, "Conditions": formatted_condition}
            for mw, tpsa in zip(mw_values, tpsa_values)
            if abs(mw - mw_mean) <= 1 * mw_std and abs(tpsa - tpsa_mean) <= 1 * tpsa_std
        ]

        all_data.extend(filtered_data)

    all_data_df = pd.DataFrame(all_data)

    # Plot the multiconditional generation distribution
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=all_data_df,
        x="Molecular Weight",
        y="TPSA",
        hue="Conditions",
        palette=colors[: len(smiles_files)],
        s=10,
    )
    plt.axvline(x=350, color="black", linestyle="--", linewidth=1, zorder=0)
    plt.axvline(x=500, color="black", linestyle="--", linewidth=1, zorder=0)
    plt.axhline(y=50, color="black", linestyle="--", linewidth=1, zorder=0)
    plt.axhline(y=120, color="black", linestyle="--", linewidth=1, zorder=0)
    plt.title(
        "Generation Conditioned on Discrete TPSA and Molecular Weight Values",
        fontsize=20,
        weight="bold",
        pad=20,
    )
    plt.xlim(240, 650)
    plt.ylim(0, 200)
    plt.grid(False)
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(save_dir, "multi_condition_generation.png"))
    plt.show()


# Define a function to plot the complete conditional generation assessment
def plot_conditional_generation_assessment(
    directory_path="results/conditional_generation_assessment",
    save_dir="results/visualization_results",
):
    """
    Plots the conditional generation assessment for a given directory of files.

    Args:
        directory_path (str): The path to the directory containing the files.
        save_dir (str): The directory where the visualization results will be saved.

    Returns:
        None
    """

    os.makedirs(save_dir, exist_ok=True)

    # Define a dictionary of descriptor information
    descriptors_dict = {
        "MW": {"name": "Molecular Weight", "index": 0, "bw": 0.3},
        "HBA": {"name": "Number of Hydrogen Bond Acceptors", "index": 1, "bw": 0.4},
        "HBD": {"name": "Number of Hydrogen Bond Donors", "index": 2, "bw": 0.5},
        "nRot": {"name": "Number of Rotatable Bonds", "index": 3, "bw": 0.3},
        "nRing": {"name": "Number of Rings", "index": 4, "bw": 0.85},
        "nHet": {"name": "Number of Heteroatoms", "index": 5, "bw": 0.3},
        "TPSA": {"name": "Topological Polar Surface Area", "index": 7, "bw": 0.4},
        "LogP": {
            "name": "Logarithm of the Partition Coefficient",
            "index": 8,
            "bw": 0.3,
        },
        "StereoCenters": {"name": "Number of Stereocenters", "index": 9, "bw": 0.4},
    }

    files = [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if "_" not in f
    ]

    # Create list of ADMET properties
    descriptor_keys = list(descriptors_dict.keys())

    _, axes = plt.subplots(3, 3, figsize=(22, 14))
    axes = axes.flatten()

    for i, descriptor_key in enumerate(descriptor_keys):

        # Obtain the relevant files for the given ADMET property
        relevant_files = [f for f in files if descriptor_key.lower() in f.lower()]
        relevant_files.sort(
            key=lambda x: int("".join(filter(str.isdigit, os.path.basename(x))))
        )

        # Plot the conditional generation distribution
        plot_conditional_generation_distribution(
            descriptors_dict, axes[i], relevant_files, descriptor_key
        )

    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(save_dir, "single_condition_generation.png"))
    plt.show()

    # Plot the multiconditional generation distribution
    plot_multiconditional_generation_distribution(directory_path, save_dir)


# Define a function to analyze the FDA compound predictions
def analyze_FDA_compound_predictions(device="gpu", save_dir="results/visualization_results"):
    """
    Analyzes the predicted activity of FDA-approved drugs on cardiac ion channels and generates visualizations.

    Args:
        device (str, optional): The device to use for prediction. Defaults to 'gpu'.
        save_dir (str, optional): The directory to save the visualization results. Defaults to 'visualization_results'.

    Returns:
        None
    """

    os.makedirs(save_dir, exist_ok=True)

    # Supress warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Predict the class of FDA-approved drugs with respect to hERG, Nav1.5, and Cav1.2 channels
    data = screen_FDA_compounds(
        prediction_type="classification",
        predict_hERG=True,
        predict_Nav=True,
        predict_Cav=True,
        save=False,
        device=device,
    )

    # Define the total number of unique compounds
    total_drugs = data["SMILES"].nunique()

    # Define the set of compounds that block hERG, Nav1.5, and Cav1.2 channels
    herg_blockers = set(data[data["hERG"] == 1.0]["SMILES"])
    nav_blockers = set(data[data["NaV1.5"] == 1.0]["SMILES"])
    cav_blockers = set(data[data["CaV1.2"] == 1.0]["SMILES"])

    # Plot the Venn diagram of the FDA-approved drug predictions with respect to the cardiac ion channels
    plt.figure(figsize=(8, 8))
    venn3(
        subsets=(
            len(herg_blockers),
            len(nav_blockers),
            len(herg_blockers & nav_blockers),
            len(cav_blockers),
            len(herg_blockers & cav_blockers),
            len(nav_blockers & cav_blockers),
            len(herg_blockers & nav_blockers & cav_blockers),
        ),
        set_labels=(
            f"hERG Blockers\n({len(herg_blockers)/total_drugs:.1%} of approved drugs)",
            f"Nav Blockers\n({len(nav_blockers)/total_drugs:.1%} of approved drugs)",
            f"Cav Blockers\n({len(cav_blockers)/total_drugs:.1%} of approved drugs)",
        ),
    )
    plt.title(
        f"Overlap of Predicted Activity of FDA-Approved Drugs on Cardiac Ion Channels\n(Total Dataset: {total_drugs})",
        fontweight="bold",
    )
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(save_dir, "fda_drug_predictions_venn.png"))
    plt.show()

    # Define the categories of predicted FDA-approved drug interactions with the cardiac ion channels
    categories = {
        "Only hERG": len(herg_blockers - nav_blockers - cav_blockers),
        "Only Nav": len(nav_blockers - herg_blockers - cav_blockers),
        "Only Cav": len(cav_blockers - herg_blockers - nav_blockers),
        "hERG & Nav": len(herg_blockers & nav_blockers - cav_blockers),
        "hERG & Cav": len(herg_blockers & cav_blockers - nav_blockers),
        "Nav & Cav": len(nav_blockers & cav_blockers - herg_blockers),
        "hERG & Nav & Cav": len(herg_blockers & nav_blockers & cav_blockers),
        "Total hERG": len(herg_blockers),
        "Total Nav": len(nav_blockers),
        "Total Cav": len(cav_blockers),
    }

    # Calculate the percentage of total compounds in each category
    percentage_categories = {k: (v / total_drugs) * 100 for k, v in categories.items()}

    # Plot the distribution of predicted FDA-approved drug interactions with the cardiac ion channels
    _, ax = plt.subplots(figsize=(12, 8))
    bars = plt.bar(
        percentage_categories.keys(), percentage_categories.values(), color="skyblue"
    )
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            f"{yval:.1f}%",
            ha="center",
            va="bottom",
        )
    ax.set_ylabel("Percentage of Total Drugs (%)")
    ax.set_title(
        "Distribution of Predicted FDA-Approved Drug Interactions with Cardiac Ion Channels",
        fontweight="bold",
    )
    ax.set_xticklabels(percentage_categories.keys(), rotation=45)
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(save_dir, "fda_drug_predictions_distribution.png"))
    plt.show()
