# Import necessary libraries
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import auc, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import GATConv, global_add_pool
from torch_geometric.utils import add_self_loops

from src.Discriminator import Discriminator_Dataset
from src.utils import calculate_classification_metrics, predict_for_channel


# Define a class for assessing different feature combinations for the discriminator model
class Feature_Variable_Discriminator(torch.nn.Module):
    """
    A PyTorch module that represents a feature variable discriminator.

    Args:
        n_graph_feats (int): The number of graph features.
        fingerprint_size (int): The size of the fingerprint.
        transformer_vector_size (int): The size of the transformer feature vector.
        gat_out_dim (int): The output dimension of the GAT layers.
        n_gat_heads (int): The number of attention heads in the GAT layers.
        dropout (float): The dropout rate.
        output_dim (int): The output dimension of the discriminator.
        device (str): The device to run the discriminator on.
        graph (bool): Whether to use the graph part of the discriminator.
        fp (bool): Whether to use the fingerprint part of the discriminator.
        tfv (bool): Whether to use the transformer feature vector part of the discriminator.
        batch_norm (bool): Whether to use batch normalization.

    Attributes:
        device (str): The device to run the discriminator on.
        graph (bool): Whether to use the graph part of the discriminator.
        fp (bool): Whether to use the fingerprint part of the discriminator.
        tfv (bool): Whether to use the transformer feature vector part of the discriminator.
        batch_norm (bool): Whether to use batch normalization.
        gat1 (GATConv): The first GAT layer.
        gat2 (GATConv): The second GAT layer.
        fc1 (nn.Linear): The first fully connected layer for the fingerprint.
        bn1 (nn.BatchNorm1d): The batch normalization layer for the first fully connected layer.
        dropout1 (nn.Dropout): The dropout layer for the first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer for the fingerprint.
        bn2 (nn.BatchNorm1d): The batch normalization layer for the second fully connected layer.
        dropout2 (nn.Dropout): The dropout layer for the second fully connected layer.
        fc3 (nn.Linear): The first fully connected layer for the transformer feature vector.
        bn3 (nn.BatchNorm1d): The batch normalization layer for the first fully connected layer.
        dropout3 (nn.Dropout): The dropout layer for the first fully connected layer.
        fc4 (nn.Linear): The second fully connected layer for the transformer feature vector.
        bn4 (nn.BatchNorm1d): The batch normalization layer for the second fully connected layer.
        dropout4 (nn.Dropout): The dropout layer for the second fully connected layer.
        combined_linear1_gat_fp_tfv (nn.Linear): The first combined linear layer for graph, fingerprint, and transformer feature vector.
        bn_combined_gat_fp_tfv (nn.BatchNorm1d): The batch normalization layer for the first combined linear layer.
        dropout_combined_gat_fp_tfv (nn.Dropout): The dropout layer for the first combined linear layer.
        combined_linear2_gat_fp_tfv (nn.Linear): The second combined linear layer for graph, fingerprint, and transformer feature vector.
        combined_linear1_gat_fp (nn.Linear): The first combined linear layer for graph and fingerprint.
        bn_combined_gat_fp (nn.BatchNorm1d): The batch normalization layer for the first combined linear layer.
        dropout_combined_gat_fp (nn.Dropout): The dropout layer for the first combined linear layer.
        combined_linear2_gat_fp (nn.Linear): The second combined linear layer for graph and fingerprint.
        combined_linear1_fp_tfv (nn.Linear): The first combined linear layer for fingerprint and transformer feature vector.
        bn_combined_fp_tfv (nn.BatchNorm1d): The batch normalization layer for the first combined linear layer.
        dropout_combined_fp_tfv (nn.Dropout): The dropout layer for the first combined linear layer.
        combined_linear2_fp_tfv (nn.Linear): The second combined linear layer for fingerprint and transformer feature vector.
        combined_linear1_gat_tfv (nn.Linear): The first combined linear layer for graph and transformer feature vector.
        bn_combined_gat_tfv (nn.BatchNorm1d): The batch normalization layer for the first combined linear layer.
        dropout_combined_gat_tfv (nn.Dropout): The dropout layer for the first combined linear layer.
        combined_linear2_gat_tfv (nn.Linear): The second combined linear layer for graph and transformer feature vector.
        combined_linear1_gat (nn.Linear): The first combined linear layer for graph.
        bn_combined_gat (nn.BatchNorm1d): The batch normalization layer for the first combined linear layer.
        dropout_combined_gat (nn.Dropout): The dropout layer for the first combined linear layer.
        combined_linear2_gat (nn.Linear): The second combined linear layer for graph.
        combined_linear1_fp (nn.Linear): The first combined linear layer for fingerprint.
        bn_combined_fp (nn.BatchNorm1d): The batch normalization layer for the first combined linear layer.
        dropout_combined_fp (nn.Dropout): The dropout layer for the first combined linear layer.
        combined_linear2_fp (nn.Linear): The second combined linear layer for fingerprint.
        combined_linear1_tfv (nn.Linear): The first combined linear layer for transformer feature vector.
        bn_combined_tfv (nn.BatchNorm1d): The batch normalization layer for the first combined linear layer.
        dropout_combined_tfv (nn.Dropout): The dropout layer for the first combined linear layer.
        combined_linear2_tfv (nn.Linear): The second combined linear layer for transformer feature vector.

    Methods:
        forward(graph_data, fingerprint, transformer_features):
            Performs forward pass through the discriminator.

    Returns:
        torch.Tensor: The output prediction tensor.
    """

    # Define the constructor
    def __init__(
        self,
        n_graph_feats=14,
        fingerprint_size=1024,
        transformer_vector_size=256,
        gat_out_dim=32,
        n_gat_heads=1,
        dropout=0.5,
        output_dim=1,
        device="gpu",
        graph=False,
        fp=False,
        tfv=False,
        batch_norm=True,
    ):
        super(Feature_Variable_Discriminator, self).__init__()

        self.device = device
        self.graph = graph
        self.fp = fp
        self.tfv = tfv
        self.batch_norm = batch_norm

        # GAT part for graph data
        self.gat1 = GATConv(n_graph_feats, n_graph_feats, heads=n_gat_heads)
        self.gat2 = GATConv(n_graph_feats * n_gat_heads, gat_out_dim, heads=n_gat_heads)

        # Fully connected part for fingerprint
        self.fc1 = nn.Linear(fingerprint_size, 400)
        self.bn1 = nn.BatchNorm1d(400)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(400, 200)
        self.bn2 = nn.BatchNorm1d(200)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        self.dropout2 = nn.Dropout(dropout)

        # Fully connected part for transformer feature vector
        self.fc3 = nn.Linear(transformer_vector_size, 400)
        self.bn3 = nn.BatchNorm1d(400)
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity="relu")
        self.dropout3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(400, 200)
        self.bn4 = nn.BatchNorm1d(200)
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity="relu")
        self.dropout4 = nn.Dropout(dropout)

        # Combined output layers for graph and fp
        combined_size_gat_fp = gat_out_dim * n_gat_heads + 200
        self.combined_linear1_gat_fp = nn.Linear(combined_size_gat_fp, 100)
        self.bn_combined_gat_fp = nn.BatchNorm1d(100)
        self.dropout_combined_gat_fp = nn.Dropout(dropout)
        self.combined_linear2_gat_fp = nn.Linear(100, output_dim)

        # Combined output layers for fp and tfv
        combined_size_fp_tfv = 200 + 200
        self.combined_linear1_fp_tfv = nn.Linear(combined_size_fp_tfv, 100)
        self.bn_combined_fp_tfv = nn.BatchNorm1d(100)
        self.dropout_combined_fp_tfv = nn.Dropout(dropout)
        self.combined_linear2_fp_tfv = nn.Linear(100, output_dim)

        # Combined output layers for graph and tfv
        combined_size_gat_tfv = gat_out_dim * n_gat_heads + 200
        self.combined_linear1_gat_tfv = nn.Linear(combined_size_gat_tfv, 100)
        self.bn_combined_gat_tfv = nn.BatchNorm1d(100)
        self.dropout_combined_gat_tfv = nn.Dropout(dropout)
        self.combined_linear2_gat_tfv = nn.Linear(100, output_dim)

        # Combied output layers for graph, fp and tfv
        combined_size_gat_fp_tfv = gat_out_dim * n_gat_heads + 200 + 200
        self.combined_linear1_gat_fp_tfv = nn.Linear(combined_size_gat_fp_tfv, 100)
        self.bn_combined_gat_fp_tfv = nn.BatchNorm1d(100)
        self.dropout_combined_gat_fp_tfv = nn.Dropout(dropout)
        self.combined_linear2_gat_fp_tfv = nn.Linear(100, output_dim)

        # Final output layer for graph
        combined_size_gat = gat_out_dim * n_gat_heads
        self.combined_linear1_gat = nn.Linear(combined_size_gat, 100)
        self.bn_combined_gat = nn.BatchNorm1d(100)
        self.dropout_combined_gat = nn.Dropout(dropout)
        self.combined_linear2_gat = nn.Linear(100, output_dim)

        # Final output layer for fp
        self.combined_linear1_fp = nn.Linear(200, 100)
        self.bn_combined_fp = nn.BatchNorm1d(100)
        self.dropout_combined_fp = nn.Dropout(dropout)
        self.combined_linear2_fp = nn.Linear(100, output_dim)

        # Final output layer for tfv
        self.combined_linear1_tfv = nn.Linear(200, 100)
        self.bn_combined_tfv = nn.BatchNorm1d(100)
        self.dropout_combined_tfv = nn.Dropout(dropout)
        self.combined_linear2_tfv = nn.Linear(100, output_dim)

    # Define the forward method
    def forward(self, graph_data=None, fingerprint=None, transformer_features=None):

        if self.graph:
            graph_data.edge_index, graph_data.edge_attr = add_self_loops(
                graph_data.edge_index, graph_data.edge_attr.view(-1)
            )
            node_feat = F.relu(self.gat1(graph_data.x, graph_data.edge_index))
            node_feat = F.relu(self.gat2(node_feat, graph_data.edge_index))
            pool_x = global_add_pool(node_feat, graph_data.batch)

        if self.fp:
            fp_out = (
                F.relu(self.bn1(self.fc1(fingerprint)))
                if self.batch_norm
                else F.relu(self.fc1(fingerprint))
            )
            fp_out = self.dropout1(fp_out)
            fp_out = (
                F.relu(self.bn2(self.fc2(fp_out)))
                if self.batch_norm
                else F.relu(self.fc2(fp_out))
            )
            fp_out = self.dropout2(fp_out)

        if self.tfv:
            tfv_out = (
                F.relu(self.bn3(self.fc3(transformer_features)))
                if self.batch_norm
                else F.relu(self.fc3(transformer_features))
            )
            tfv_out = self.dropout3(tfv_out)
            tfv_out = (
                F.relu(self.bn4(self.fc4(tfv_out)))
                if self.batch_norm
                else F.relu(self.fc4(tfv_out))
            )
            tfv_out = self.dropout4(tfv_out)

        if self.graph and self.fp and self.tfv:
            combined = (
                torch.cat([pool_x, fp_out, tfv_out], dim=1)
                if self.batch_norm
                else torch.cat(
                    [pool_x, fp_out.unsqueeze(0), tfv_out.unsqueeze(0)], dim=1
                )
            )
            combined = (
                F.relu(
                    self.bn_combined_gat_fp_tfv(
                        self.combined_linear1_gat_fp_tfv(combined)
                    )
                )
                if self.batch_norm
                else F.relu(self.combined_linear1_gat_fp_tfv(combined))
            )
            combined = self.dropout_combined_gat_fp_tfv(combined)
            prediction = self.combined_linear2_gat_fp_tfv(combined)

        elif self.graph and self.fp:
            combined = torch.cat([pool_x, fp_out], dim=1)
            combined = (
                F.relu(self.bn_combined_gat_fp(self.combined_linear1_gat_fp(combined)))
                if self.batch_norm
                else F.relu(self.combined_linear1_gat_fp(combined))
            )
            combined = self.dropout_combined_gat_fp(combined)
            prediction = self.combined_linear2_gat_fp(combined)

        elif self.fp and self.tfv:
            combined = torch.cat([fp_out, tfv_out], dim=1)
            combined = (
                F.relu(self.bn_combined_fp_tfv(self.combined_linear1_fp_tfv(combined)))
                if self.batch_norm
                else F.relu(self.combined_linear1_fp_tfv(combined))
            )
            combined = self.dropout_combined_fp_tfv(combined)
            prediction = self.combined_linear2_fp_tfv(combined)

        elif self.graph and self.tfv:
            combined = torch.cat([pool_x, tfv_out], dim=1)
            combined = (
                F.relu(
                    self.bn_combined_gat_tfv(self.combined_linear1_gat_tfv(combined))
                )
                if self.batch_norm
                else F.relu(self.combined_linear1_gat_tfv(combined))
            )
            combined = self.dropout_combined_gat_tfv(combined)
            prediction = self.combined_linear2_gat_tfv(combined)

        elif self.graph:
            pool_x = (
                F.relu(self.bn_combined_gat(self.combined_linear1_gat(pool_x)))
                if self.batch_norm
                else F.relu(self.combined_linear1_gat(pool_x))
            )
            pool_x = self.dropout_combined_gat(pool_x)
            prediction = self.combined_linear2_gat(pool_x)

        elif self.fp:
            fp_out = (
                F.relu(self.bn_combined_fp(self.combined_linear1_fp(fp_out)))
                if self.batch_norm
                else F.relu(self.combined_linear1_fp(fp_out))
            )
            fp_out = self.dropout_combined_fp(fp_out)
            prediction = self.combined_linear2_fp(fp_out)

        elif self.tfv:
            tfv_out = (
                F.relu(self.bn_combined_tfv(self.combined_linear1_tfv(tfv_out)))
                if self.batch_norm
                else F.relu(self.combined_linear1_tfv(tfv_out))
            )
            tfv_out = self.dropout_combined_tfv(tfv_out)
            prediction = self.combined_linear2_tfv(tfv_out)

        return prediction


# Define a function to evaluate the feature variable discriminator model
def evaluate_feature_variable_discriminator_model(
    device="gpu",
    test_data=None,
    model_params=None,
    graph_bool=False,
    fp_bool=False,
    tfv_bool=False,
    return_logits=False,
):
    """
    Evaluate the feature variable discriminator model on test data.

    Args:
        device (str): The device to use for evaluation. Must be one of ['cpu', 'gpu'].
        test_data (list): The test data to evaluate the model on.
        model_params (str): The file path to the trained model parameters.
        graph_bool (bool): Whether to include graph feature representation.
        fp_bool (bool): Whether to include fingerprint feature representation.
        tfv_bool (bool): Whether to include transformer feature vector representation.
        return_logits (bool): Whether to return the logits in addition to predictions.

    Returns:
        pandas.DataFrame: A DataFrame containing the classification results.

    Raises:
        AssertionError: If an invalid device value is provided.

    """

    assert device in [
        "cpu",
        "gpu",
    ], "Invalid device value. Must be one of ['cpu', 'gpu']"

    device = torch.device(
        "cuda:0" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
    )

    # Define and prepare the model
    model = Feature_Variable_Discriminator(
        graph=graph_bool, fp=fp_bool, tfv=tfv_bool, device=device
    ).float()
    model_train_dict = torch.load(model_params, map_location=device)
    model.load_state_dict(model_train_dict["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    # Load the test data
    dataset = Discriminator_Dataset(test_data, classification=True)
    dataloader = DataListLoader(dataset, batch_size=32, shuffle=False, drop_last=False)

    feature_representations = []
    if graph_bool:
        feature_representations.append("Graph")
    if fp_bool:
        feature_representations.append("Fingerprint")
    if tfv_bool:
        feature_representations.append("Transformer Feature Vector")

    feature_representation = " + ".join(feature_representations)

    # Evaluate the model on the test data
    _, predictions, labels = predict_for_channel(
        dataloader,
        "classification",
        model,
        feature_representation,
        device,
        labels=True,
        logits=return_logits,
    )

    # Calculate classification metrics
    classification_results = []
    AC, SN, SP, F1, CCR, MCC = calculate_classification_metrics(predictions, labels)
    classification_results.append(
        {
            "Feature Representation": feature_representation,
            "Accuracy (AC)": float(AC),
            "Sensitivity (SN)": float(SN),
            "Specificity (SP)": float(SP),
            "F1-score (F1)": float(F1),
            "Correct Classification Rate (CCR)": float(CCR),
            "Matthew’s Correlation Coefficient (MCC)": float(MCC),
        }
    )

    # Return the classification results as a DataFrame
    df_classification_results = pd.DataFrame(classification_results)

    return df_classification_results


# Define a function to evaluate all feature combinations for the discriminator model
def evaluate_feature_variable_discriminator_models(
    parameters_directory="model_parameters/feature_variable_discriminator_parameters",
    test_data="data/prepared_cardiac_datasets/test_hERG.h5",
    save_dir="results/discriminative_feature_selection_analysis/",
    device="gpu",
):
    """
    Evaluates multiple feature variable discriminator models and saves the classification metrics.

    Parameters:
    - parameters_directory (str): The directory path where the model parameters are stored.
    - test_data (str): The file path of the test data.
    - save_dir (str): The directory path where the classification metrics will be saved.
    - device (str): The device to use for model evaluation (e.g., 'cpu', 'cuda').

    Returns:
    - all_results (DataFrame): The concatenated results of all evaluated models, sorted by accuracy.

    """

    os.makedirs(save_dir, exist_ok=True)

    # Load the model parameters and evaluate each model
    model_files = [f for f in os.listdir(parameters_directory) if f.endswith(".pt")]
    all_results = pd.DataFrame()
    for model_file in model_files:
        graph_bool = "Graph" in model_file
        fp_bool = "FP" in model_file
        tfv_bool = "TFV" in model_file
        results_df = evaluate_feature_variable_discriminator_model(
            device=device,
            test_data=test_data,
            model_params=os.path.join(parameters_directory, model_file),
            graph_bool=graph_bool,
            fp_bool=fp_bool,
            tfv_bool=tfv_bool,
        )
        all_results = pd.concat([all_results, results_df], axis=0)

    # Save the classification metrics to a CSV file
    all_results = all_results.sort_values(by="Accuracy (AC)", ascending=False)
    all_results = all_results.map(
        lambda x: f"{x*100:.1f}" if isinstance(x, float) else x
    )
    all_results = all_results.reset_index(drop=True)
    all_results.to_csv(
        os.path.join(save_dir, "classification_metrics.csv"), index=False
    )
    print(f"Classification metrics saved to {save_dir}classification_metrics.csv. ✅")

    return all_results[:]


# Define a function to plot the ROC curve for different feature combinations
def plot_roc_curve_for_different_feature_combinations(
    parameters_directory="model_parameters/feature_variable_discriminator_parameters",
    test_data="data/prepared_cardiac_datasets/test_hERG.h5",
    device="gpu",
    save_dir="results/discriminative_feature_selection_analysis/",
):
    """
    Plots the Receiver Operating Characteristic (ROC) curve for different feature combinations.

    Args:
        parameters_directory (str): The directory path where the model parameters are stored.
        test_data (str): The path to the test data file.
        device (str): The device to use for model evaluation ('cpu' or 'gpu').
        save_dir (str): The directory path to save the generated ROC curve plot.

    Returns:
        None
    """

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(
        "cuda:0" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
    )

    # Retrieve the model parameters
    model_files = [f for f in os.listdir(parameters_directory) if f.endswith(".pt")]

    # Load the test data
    dataset = Discriminator_Dataset(test_data, classification=True)
    dataloader = DataListLoader(dataset, batch_size=32, shuffle=False, drop_last=False)

    # Evaluate each model on the test data
    all_results = []
    for model_file in model_files:
        model_path = os.path.join(parameters_directory, model_file)
        graph_bool = "Graph" in model_file
        fp_bool = "FP" in model_file
        tfv_bool = "TFV" in model_file
        feature_representations = []
        if graph_bool:
            feature_representations.append("Graph")
        if fp_bool:
            feature_representations.append("Fingerprint")
        if tfv_bool:
            feature_representations.append("Transformer Feature Vector")

        feature_representation = " + ".join(feature_representations)

        # Load the model
        model = Feature_Variable_Discriminator(
            graph=graph_bool, fp=fp_bool, tfv=tfv_bool, device=device
        ).float()
        model_train_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(model_train_dict["model_state_dict"], strict=False)
        model.to(device)
        model.eval()

        # Evaluate the model on the test data
        _, probabilities, labels = predict_for_channel(
            dataloader,
            "classification",
            model,
            feature_representation,
            device,
            labels=True,
            logits=True,
        )

        # Calculate the ROC curve and AUC
        fpr, tpr, _ = roc_curve(labels, probabilities)
        roc_auc = auc(fpr, tpr)
        all_results.append(
            (feature_representation, {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc})
        )
        all_results.sort(key=lambda x: x[1]["roc_auc"], reverse=True)

    # Plot the ROC curve for each feature representation
    plt.figure(figsize=(15, 9))
    colors = [
        "#E57373",
        "#FFB74D",
        "#FFF176",
        "#81C784",
        "#64B5F6",
        "#9575CD",
        "#F06292",
    ]
    for i, (feature_representation, results) in enumerate(all_results):
        plt.plot(
            results["fpr"],
            results["tpr"],
            color=colors[i],
            lw=2,
            label=f'{feature_representation} (AUC = {results["roc_auc"]:.2f})',
        )
    plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--", label="Random Chance")
    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive Rate", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(
        "Receiver Operating Characteristic for Each Test Set",
        fontsize=20,
        weight="bold",
    )
    plt.legend(loc="lower right", fontsize=16, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=300)
    plt.show()
