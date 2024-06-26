# Import necessary libraries
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
from openbabel import openbabel, pybel
import os
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import auc, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import GATConv, global_add_pool
from torch_geometric.utils import add_self_loops
from tqdm import tqdm

from src.Transformer import Transformer_Feature_Extractor
from src.utils import (
    calculate_classification_metrics,
    decode_smiles,
    get_fingerprint_features,
    get_graph_features,
    l1_regularization,
    predict_for_channel,
    validate,
)


# Define a class for the discriminator dataset
class Discriminator_Dataset(Dataset):
    """
    Dataset class for the discriminator model.

    Args:
        input_data (str): Path to the input data file.
        classification (bool, optional): Whether the dataset is for classification or regression.
            Defaults to True.

    Attributes:
        keys (list): List of keys in the input data file.
        input_data (str): Path to the input data file.
        classification (bool): Whether the dataset is for classification or regression.

    """

    # Define the class constructor
    def __init__(self, input_data, classification=True):
        super(Discriminator_Dataset, self).__init__()

        # Retrieve all encoded SMILES keys from the input data file
        with h5py.File(input_data, "r") as file:
            self.keys = list(file.keys())

        self.input_data = input_data
        self.classification = classification

    # Method to get the length of the dataset
    def __len__(self):
        return len(self.keys)

    # Method to get a sample from the dataset
    def __getitem__(self, idx):

        with h5py.File(self.input_data, "r") as data:
            # Retrieve the encoded SMILES key
            smiles_key = self.keys[idx]
            data_group = data[smiles_key]

            # Define the label
            if self.classification:
                label = 1 if data_group.attrs["label"] >= 5.0 else 0
            else:
                label = data_group.attrs["label"]

            # Retrieve the graph, fingerprint, and transformer features
            graph_feats = data_group["graph"][:]
            fp_feats = data_group["fingerprint"][:]
            transformer_feats = data_group["transformer_vector"][:]

            # Decode the SMILES string
            decoded_smiles = decode_smiles(smiles_key)

            # Calculate the edge indices and edge features
            mol = pybel.readstring("smi", decoded_smiles)
            edge_inds, edge_feats = [], []
            for bond in openbabel.OBMolBondIter(mol.OBMol):
                i, j = bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1
                bond_order = bond.GetBondOrder()
                edge_inds.append((i, j))
                edge_inds.append((j, i))
                edge_feats += [bond_order, bond_order]
            edge_inds = torch.tensor(edge_inds, dtype=torch.long).t().contiguous()
            edge_feats = torch.tensor(edge_feats, dtype=torch.float).view(-1, 1)

            # Create and return the data object
            data = Data(
                x=torch.from_numpy(graph_feats).float(),
                edge_index=edge_inds,
                edge_attr=edge_feats,
                fingerprint=torch.from_numpy(fp_feats).float(),
                transformer_vector=transformer_feats,
                label=torch.FloatTensor([label]),
                smiles=decoded_smiles,
            )

            return data


# Define the discriminator model
class Discriminator_Model(torch.nn.Module):
    """
    Discriminator model for graph-based classification.

    Args:
        n_graph_feats (int): Number of graph features.
        fingerprint_size (int): Size of the fingerprint.
        transformer_vector_size (int): Size of the transformer feature vector.
        gat_out_dim (int): Output dimension of the GAT layers.
        n_gat_heads (int): Number of attention heads in the GAT layers.
        dropout (float): Dropout rate.
        device (str): Device to run the model on.
        batch_norm (bool): Whether to use batch normalization.

    Attributes:
        gat1 (GATConv): First GAT layer for graph processing.
        gat2 (GATConv): Second GAT layer for graph processing.
        fc1 (nn.Linear): First fully connected layer for fingerprint processing.
        bn1 (nn.BatchNorm1d): Batch normalization layer for the first fully connected layer.
        dropout1 (nn.Dropout): Dropout layer for the first fully connected layer.
        fc2 (nn.Linear): Second fully connected layer for fingerprint processing.
        bn2 (nn.BatchNorm1d): Batch normalization layer for the second fully connected layer.
        dropout2 (nn.Dropout): Dropout layer for the second fully connected layer.
        fc3 (nn.Linear): First fully connected layer for transformer feature vector processing.
        bn3 (nn.BatchNorm1d): Batch normalization layer for the first fully connected layer.
        dropout3 (nn.Dropout): Dropout layer for the first fully connected layer.
        fc4 (nn.Linear): Second fully connected layer for transformer feature vector processing.
        bn4 (nn.BatchNorm1d): Batch normalization layer for the second fully connected layer.
        dropout4 (nn.Dropout): Dropout layer for the second fully connected layer.
        combined_linear1_gat_fp_tfv (nn.Linear): First fully connected layer for combined output.
        bn_combined_gat_fp_tfv (nn.BatchNorm1d): Batch normalization layer for the combined output.
        dropout_combined_gat_fp_tfv (nn.Dropout): Dropout layer for the combined output.
        combined_linear2_gat_fp_tfv (nn.Linear): Second fully connected layer for combined output.

    """

    # Define the class constructor
    def __init__(
        self,
        n_graph_feats=14,
        fingerprint_size=1024,
        transformer_vector_size=256,
        gat_out_dim=32,
        n_gat_heads=1,
        dropout=0.5,
        device="gpu",
        batch_norm=True,
    ):

        super(Discriminator_Model, self).__init__()
        self.device = device
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

        # Combined output layers
        combined_size_gat_fp_tfv = gat_out_dim * n_gat_heads + 200 + 200
        self.combined_linear1_gat_fp_tfv = nn.Linear(combined_size_gat_fp_tfv, 100)
        self.bn_combined_gat_fp_tfv = nn.BatchNorm1d(100)
        self.dropout_combined_gat_fp_tfv = nn.Dropout(dropout)
        self.combined_linear2_gat_fp_tfv = nn.Linear(100, 1)

    # Method to forward pass the input data through the model
    def forward(self, graph_data=None, fingerprint=None, transformer_features=None):

        # Process the graph data
        graph_data.edge_index, graph_data.edge_attr = add_self_loops(
            graph_data.edge_index, graph_data.edge_attr.view(-1)
        )
        graph_feat = F.relu(self.gat1(graph_data.x, graph_data.edge_index))
        graph_feat = F.relu(self.gat2(graph_feat, graph_data.edge_index))
        pool_x = global_add_pool(graph_feat, graph_data.batch)

        # Process the fingerprint
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

        # Process the transformer feature vector
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

        # Combine the outputs
        combined = (
            torch.cat([pool_x, fp_out, tfv_out], dim=1)
            if self.batch_norm
            else torch.cat([pool_x, fp_out.unsqueeze(0), tfv_out.unsqueeze(0)], dim=1)
        )
        combined = (
            F.relu(
                self.bn_combined_gat_fp_tfv(self.combined_linear1_gat_fp_tfv(combined))
            )
            if self.batch_norm
            else F.relu(self.combined_linear1_gat_fp_tfv(combined))
        )
        combined = self.dropout_combined_gat_fp_tfv(combined)

        # Final prediction
        prediction = self.combined_linear2_gat_fp_tfv(combined)

        return prediction


# Define a function to train the discriminator model
def train_discriminator(
    training_data,
    validation_data=None,
    checkpoint_path=None,
    classification=True,
    epochs=100,
    batch_size=32,
    learning_rate=3e-4,
    weight_decay=1e-4,
    device="gpu",
    verbose=False,
):
    """
    Trains the discriminator model using the provided training data.

    Args:
        training_data (list): List of training data.
        validation_data (list, optional): List of validation data. Defaults to None.
        checkpoint_path (str, optional): Path to save the model checkpoints. Defaults to None.
        classification (bool, optional): Whether the task is classification or regression. Defaults to True.
        epochs (int, optional): Number of training epochs. Defaults to 100.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 3e-4.
        weight_decay (float, optional): Weight decay for the optimizer. Defaults to 1e-4.
        device (str, optional): Device to use for training ('cpu' or 'gpu'). Defaults to 'cpu'.
        verbose (bool, optional): Whether to print training progress. Defaults to False.

    Returns:
        None
    """

    assert device in ["cpu", "gpu"], "Device must be either 'cpu' or 'gpu'."
    device = torch.device(
        "cuda:0" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
    )

    # Load the training and validation datasets
    training_dataset = Discriminator_Dataset(
        training_data, classification=classification
    )
    validation_dataset = Discriminator_Dataset(
        validation_data, classification=classification
    )
    training_dataloader = DataListLoader(
        training_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    validation_dataloader = DataListLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    (
        print(
            "Successfully loaded {} training samples and {} validation samples.".format(
                len(training_dataset), len(validation_dataset)
            )
        )
        if verbose
        else None
    )

    # Build the model
    model = Discriminator_Model(device=device).float()
    model.train()
    model.to(device)
    print("Successfully built model.") if verbose else None

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=10, factor=0.5, verbose=True
    )
    loss_function = (
        torch.nn.BCEWithLogitsLoss() if classification else torch.nn.MSELoss()
    )

    if classification:
        best_val_accuracy = float("-inf")
        (
            epoch_train_losses,
            epoch_val_losses,
            epoch_train_accuracy,
            epoch_val_accuracy,
        ) = ([], [], [], [])

    else:
        best_val_correlation = float("-inf")
        (
            epoch_train_losses,
            epoch_val_losses,
            epoch_train_correlation,
            epoch_val_correlation,
        ) = ([], [], [], [])

    print("Beginning training...") if verbose else None
    (
        print(
            "------------------------------------------------------------------------------------------------------------------"
        )
        if verbose
        else None
    )

    # Train the model
    for epoch in range(1, epochs + 1):

        epoch_loss_accumulator = 0
        epoch_score_accumulator = 0
        total_batches = 0

        with tqdm(
            enumerate(training_dataloader),
            total=len(training_dataloader),
            desc=f"Epoch {epoch} Training",
            leave=True,
        ) as t:
            for _, batch in t:
                try:
                    # Zero the gradients
                    optimizer.zero_grad()

                    # Retrieve the batch data
                    (
                        graph_data_list,
                        fingerprint_list,
                        transformer_vector_list,
                        labels,
                    ) = ([], [], [], [])
                    for data_obj in batch:
                        graph_data_list.append(data_obj)
                        fingerprint_list.append(data_obj.fingerprint)
                        transformer_vector_list.append(
                            torch.tensor(data_obj.transformer_vector, dtype=torch.float)
                        )
                        labels.append(data_obj.label)

                    batch_graph_data = Batch.from_data_list(graph_data_list).to(device)
                    batch_fp_data = torch.stack(fingerprint_list).to(device)
                    batch_tfv_data = torch.stack(transformer_vector_list).to(device)

                    # Feed the data through the model and make predictions
                    y_pred = model(
                        graph_data=batch_graph_data,
                        fingerprint=batch_fp_data,
                        transformer_features=batch_tfv_data,
                    )

                    y_true = torch.tensor(labels, dtype=torch.float).to(y_pred.device)

                    # Calculate the loss
                    loss = loss_function(y_pred.squeeze(1), y_true)
                    l1_reg = l1_regularization(model)
                    loss_l1 = loss + 1e-6 * l1_reg

                    # Backpropagation and optimization
                    loss_l1.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                    if classification:

                        # Calculate the accuracy
                        probabilities = torch.sigmoid(y_pred)
                        predictions = (probabilities > 0.5).float()
                        predictions = predictions.squeeze(1)
                        accuracy = (predictions == y_true).float().mean().item()
                        epoch_score_accumulator += accuracy
                        t.set_postfix(loss=loss_l1.item(), Accuracy=accuracy)

                    elif not classification:
                        # Calculate the Pearson correlation
                        correlation = pearsonr(
                            y_pred.squeeze(1).cpu().detach().numpy(),
                            y_true.cpu().detach().numpy(),
                        )[0]
                        epoch_score_accumulator += correlation
                        t.set_postfix(loss=loss_l1.item(), Correlation=correlation)

                    epoch_loss_accumulator += loss_l1.item()
                    total_batches += 1

                except Exception as e:
                    print(f"Error during training batch: {e}") if verbose else None
                    continue

        epoch_loss_l1 = epoch_loss_accumulator / total_batches
        epoch_score = epoch_score_accumulator / total_batches

        # Validate the model
        validation = validate(
            model=model,
            validation_dataloader=validation_dataloader,
            classification=classification,
            epoch=epoch,
            device=device,
        )

        scheduler.step(validation["loss"])

        epoch_train_losses.append(epoch_loss_l1)
        epoch_val_losses.append(validation["loss"])

        # Save the model checkpoint
        if classification:
            epoch_train_accuracy.append(epoch_score)
            epoch_val_accuracy.append(validation["score"])
            checkpoint_dict = {
                "model_state_dict": model.state_dict(),
                "training_losses": epoch_train_losses,
                "validation_losses": epoch_val_losses,
                "training_accuracy": epoch_train_accuracy,
                "validation_accuracy": epoch_val_accuracy,
                "epoch": epoch,
            }
        else:
            epoch_train_correlation.append(epoch_score)
            epoch_val_correlation.append(validation["score"])
            checkpoint_dict = {
                "model_state_dict": model.state_dict(),
                "training_losses": epoch_train_losses,
                "validation_losses": epoch_val_losses,
                "training_correlation": epoch_train_correlation,
                "validation_correlation": epoch_val_correlation,
                "epoch": epoch,
            }

        torch.save(checkpoint_dict, checkpoint_path)

        if classification:
            if checkpoint_dict["validation_accuracy"][-1] >= best_val_accuracy:
                best_val_accuracy = checkpoint_dict["validation_accuracy"][-1]
                checkpoint_dict["best_validation_accuracy"] = best_val_accuracy
                torch.save(
                    checkpoint_dict,
                    os.path.join(
                        os.path.dirname(checkpoint_path),
                        "best_" + os.path.basename(checkpoint_path),
                    ),
                )
                print(
                    "New best model saved with {:.4f} accuracy".format(
                        best_val_accuracy
                    )
                )

        else:
            if checkpoint_dict["validation_correlation"][-1] >= best_val_correlation:
                best_val_correlation = checkpoint_dict["validation_correlation"][-1]
                checkpoint_dict["best_validation_correlation"] = best_val_correlation
                torch.save(
                    checkpoint_dict,
                    os.path.join(
                        os.path.dirname(checkpoint_path),
                        "best_" + os.path.basename(checkpoint_path),
                    ),
                )
                print(
                    "New best model saved with {:.4f} correlation".format(
                        best_val_correlation
                    )
                )


# Define a class to perform inference with the discriminator model
class Discriminator_Inference(object):
    """
    A class representing the discriminator for inference.

    Args:
        device (str): The device to use for computation. Must be one of ['cpu', 'gpu'].
        bidirectional_transformer_params (dict): Parameters for the bidirectional transformer model.
        transformer_training_data (str): Path to the training data for the transformer model.
        herg_regression_params (dict): Parameters for the hERG regression model.
        nav_regression_params (dict): Parameters for the NaV1.5 regression model.
        cav_regression_params (dict): Parameters for the CaV1.2 regression model.
        herg_classification_params (dict): Parameters for the hERG classification model.
        nav_classification_params (dict): Parameters for the NaV1.5 classification model.
        cav_classification_params (dict): Parameters for the CaV1.2 classification model.
    """

    # Define the class constructor
    def __init__(
        self,
        device="gpu",
        bidirectional_transformer_params=None,
        transformer_training_data=None,
        herg_regression_params=None,
        nav_regression_params=None,
        cav_regression_params=None,
        herg_classification_params=None,
        nav_classification_params=None,
        cav_classification_params=None,
    ):

        assert device in [
            "cpu",
            "gpu",
        ], "Invalid device value. Must be one of ['cpu', 'gpu']"
        self.device = torch.device(
            "cuda:0" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
        )

        print("Loading models... (This will take about 1 minute)")

        self.transformer_training_data = transformer_training_data

        # Load the discriminative models
        self.herg_regression_model = (
            load_discriminative_model(herg_regression_params, self.device)
            if herg_regression_params
            else None
        )
        self.cav_regression_model = (
            load_discriminative_model(cav_regression_params, self.device)
            if cav_regression_params
            else None
        )
        self.nav_regression_model = (
            load_discriminative_model(nav_regression_params, self.device)
            if nav_regression_params
            else None
        )
        self.herg_classification_model = (
            load_discriminative_model(herg_classification_params, self.device)
            if herg_classification_params
            else None
        )
        self.cav_classification_model = (
            load_discriminative_model(cav_classification_params, self.device)
            if cav_classification_params
            else None
        )
        self.nav_classification_model = (
            load_discriminative_model(nav_classification_params, self.device)
            if nav_classification_params
            else None
        )

        # Load the bidirectional transformer model for feature extraction
        self.transformer_feature_extractor = Transformer_Feature_Extractor(
            model_parameters=bidirectional_transformer_params,
            training_data=self.transformer_training_data,
            device=device,
        )

    # Method to perform inference with the discriminative models
    def inference(
        self,
        input_data=None,
        prediction_type=None,
        predict_hERG=None,
        predict_Nav=None,
        predict_Cav=None,
        save_path=None,
    ):

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Prepare the data for the discriminative models
        if isinstance(input_data, str) and input_data.endswith(".h5"):
            dataset = Discriminator_Dataset(input_data)
            dataloader = DataListLoader(
                dataset, batch_size=32, shuffle=False, drop_last=False
            )

        elif isinstance(input_data, list):
            data_list = [
                prepare_data_for_discriminative_models(
                    smiles, self.transformer_feature_extractor, self.device
                )
                for smiles in input_data
            ]
            dataloader = DataListLoader(
                data_list, batch_size=32, shuffle=False, drop_last=False
            )

        elif isinstance(input_data, str):
            data = prepare_data_for_discriminative_models(
                input_data, self.transformer_feature_extractor, self.device
            )
            dataloader = DataListLoader(
                [data], batch_size=1, shuffle=False, drop_last=False
            )

        results = {}

        # Perform inference with the discriminative models
        if predict_hERG:
            if prediction_type == "classification":
                herg_smiles, herg_predictions = predict_for_channel(
                    dataloader,
                    prediction_type,
                    self.herg_classification_model,
                    "hERG",
                    self.device,
                )
            else:
                herg_smiles, herg_predictions = predict_for_channel(
                    dataloader,
                    prediction_type,
                    self.herg_regression_model,
                    "hERG",
                    self.device,
                )
            results["hERG"] = {"smiles": herg_smiles, "predictions": herg_predictions}

        if predict_Nav:
            if prediction_type == "classification":
                nav_smiles, nav_predictions = predict_for_channel(
                    dataloader,
                    prediction_type,
                    self.nav_classification_model,
                    "NaV1.5",
                    self.device,
                )
            else:
                nav_smiles, nav_predictions = predict_for_channel(
                    dataloader,
                    prediction_type,
                    self.nav_regression_model,
                    "NaV1.5",
                    self.device,
                )
            results["NaV1.5"] = {"smiles": nav_smiles, "predictions": nav_predictions}

        if predict_Cav:
            if prediction_type == "classification":
                cav_smiles, cav_predictions = predict_for_channel(
                    dataloader,
                    prediction_type,
                    self.cav_classification_model,
                    "CaV1.2",
                    self.device,
                )
            else:
                cav_smiles, cav_predictions = predict_for_channel(
                    dataloader,
                    prediction_type,
                    self.cav_regression_model,
                    "CaV1.2",
                    self.device,
                )
            results["CaV1.2"] = {"smiles": cav_smiles, "predictions": cav_predictions}

        # Save the results
        with open(save_path, "w") as f:
            json.dump(results, f)

        print(f"Results saved to {save_path} ✅")

        return results


# Define a function to prepare the data for discriminative models
def prepare_data_for_discriminative_models(
    smiles, transformer_feature_extractor, device, add_batch_dim=False
):

    # Retrieve the graph, fingerprint, and transformer features
    graph_feats = get_graph_features(smiles)
    fingerprint_feats = get_fingerprint_features(smiles)
    transformer_feats = transformer_feature_extractor.extract_features(smiles)[
        0, :
    ].flatten()

    # Calculate the edge indices and edge features
    mol = pybel.readstring("smi", smiles)
    edge_inds, edge_feats = [], []
    for bond in openbabel.OBMolBondIter(mol.OBMol):
        i = bond.GetBeginAtomIdx() - 1
        j = bond.GetEndAtomIdx() - 1
        bond_order = bond.GetBondOrder()
        edge_inds.append((i, j))
        edge_inds.append((j, i))
        edge_feats += [bond_order, bond_order]
    edge_inds = torch.tensor(edge_inds, dtype=torch.long).t().contiguous()
    edge_feats = torch.tensor(edge_feats, dtype=torch.float).view(-1, 1)

    # Create the data object
    if add_batch_dim:
        data = Data(
            x=torch.from_numpy(graph_feats).float(),
            edge_index=edge_inds,
            edge_attr=edge_feats,
            fingerprint=torch.from_numpy(fingerprint_feats).float().unsqueeze(0),
            transformer_vector=transformer_feats.unsqueeze(0),
            smiles=smiles,
        ).to(device)

    else:
        data = Data(
            x=torch.from_numpy(graph_feats).float(),
            edge_index=edge_inds,
            edge_attr=edge_feats,
            fingerprint=torch.from_numpy(fingerprint_feats).float(),
            transformer_vector=transformer_feats,
            smiles=smiles,
        ).to(device)

    return data


# Define a function to load a discriminative model
def load_discriminative_model(model_params, device):
    """
    Loads a discriminative model from the specified model parameters file.

    Args:
        model_params (str): The file path to the model parameters file.
        device (str): The device to load the model on.

    Returns:
        model: The loaded discriminative model.
    """

    # Load the model
    model = Discriminator_Model(
        n_gat_heads=1, dropout=0.5, device=device, batch_norm=True
    ).float()
    model_train_dict = torch.load(model_params, map_location=device)
    model.load_state_dict(model_train_dict["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model


# Define a function to evaluate the discriminative models
def predict_cardiac_ion_channel_activity(
    input_data="O=c1[nH]c2ccccc2n1C1CCN(CCCC(c2ccc(F)cc2)c2ccc(F)cc2)CC1",
    prediction_type="regression",
    predict_hERG=True,
    predict_Nav=False,
    predict_Cav=False,
    device="gpu",
    bidirectional_transformer_params="model_parameters/transformer_model_parameters/Bidirectional_Transformer_parameters.pt",
    transformer_training_data="data/prepared_transformer_datasets/prepared_transformer_data.csv",
    herg_regression_params="model_parameters/discriminative_model_parameters/hERG_Regression_parameters.pt",
    nav_regression_params="model_parameters/discriminative_model_parameters/Nav_Regression_parameters.pt",
    cav_regression_params="model_parameters/discriminative_model_parameters/Cav_Regression_parameters.pt",
    herg_classification_params="model_parameters/discriminative_model_parameters/hERG_Classification_parameters.pt",
    nav_classification_params="model_parameters/discriminative_model_parameters/Nav_Classification_parameters.pt",
    cav_classification_params="model_parameters/discriminative_model_parameters/Cav_Classification_parameters.pt",
    save_path="results/discriminative_results/predictions.json",
):
    """
    Predicts cardiac ion channel activity based on the input data using the discriminator model.

    Args:
        input_data (str): The input data representing the chemical structure of a compound. Default is a sample input.
        prediction_type (str): The type of prediction to perform. Default is 'regression'.
        predict_hERG (bool): Whether to predict hERG activity. Default is True.
        predict_Nav (bool): Whether to predict Nav activity. Default is False.
        predict_Cav (bool): Whether to predict Cav activity. Default is False.
        device (str): The device to use for inference. Default is 'gpu'.
        bidirectional_transformer_params (str): The file path to the bidirectional transformer model parameters.
        transformer_training_data (str): The file path to the prepared transformer training data.
        herg_regression_params (str): The file path to the hERG regression model parameters.
        nav_regression_params (str): The file path to the Nav regression model parameters.
        cav_regression_params (str): The file path to the Cav regression model parameters.
        herg_classification_params (str): The file path to the hERG classification model parameters.
        nav_classification_params (str): The file path to the Nav classification model parameters.
        cav_classification_params (str): The file path to the Cav classification model parameters.
        save_path (str): The file path to save the predictions.

    Returns:
        pandas.DataFrame: A DataFrame containing the predictions for each ion channel.
    """

    # Create inference object
    inf = Discriminator_Inference(
        device=device,
        bidirectional_transformer_params=bidirectional_transformer_params,
        transformer_training_data=transformer_training_data,
        herg_regression_params=herg_regression_params,
        nav_regression_params=nav_regression_params,
        cav_regression_params=cav_regression_params,
        herg_classification_params=herg_classification_params,
        nav_classification_params=nav_classification_params,
        cav_classification_params=cav_classification_params,
    )

    # Perform inference
    output = inf.inference(
        input_data=input_data,
        prediction_type=prediction_type,
        predict_hERG=predict_hERG,
        predict_Nav=predict_Nav,
        predict_Cav=predict_Cav,
        save_path=save_path,
    )

    # Create a DataFrame from the output and return it
    df = pd.DataFrame(output[next(iter(output))]["smiles"], columns=["smiles"])
    for key in output.keys():
        df[key] = output[key]["predictions"]

    return df[:]


# Define a function to evaluate the discriminative models
def evaluate_discriminator_model(
    prediction_type="classification",
    predict_hERG=True,
    predict_Nav=True,
    predict_Cav=True,
    device="gpu",
    herg_data="data/prepared_cardiac_datasets/test_hERG.h5",
    nav_data="data/prepared_cardiac_datasets/test_Nav.h5",
    cav_data="data/prepared_cardiac_datasets/test_Cav.h5",
    herg_regression_params="model_parameters/discriminative_model_parameters/hERG_Regression_parameters.pt",
    nav_regression_params="model_parameters/discriminative_model_parameters/Nav_Regression_parameters.pt",
    cav_regression_params="model_parameters/discriminative_model_parameters/Cav_Regression_parameters.pt",
    herg_classification_params="model_parameters/discriminative_model_parameters/hERG_Classification_parameters.pt",
    nav_classification_params="model_parameters/discriminative_model_parameters/Nav_Classification_parameters.pt",
    cav_classification_params="model_parameters/discriminative_model_parameters/Cav_Classification_parameters.pt",
    save_dir="results/discriminative_results/",
):
    """
    Evaluate the discriminator model for cardiac channel prediction.

    Args:
        prediction_type (str, optional): The type of prediction to perform. Must be one of ['regression', 'classification']. Defaults to 'regression'.
        predict_hERG (bool, optional): Whether to predict hERG channel. Defaults to True.
        predict_Nav (bool, optional): Whether to predict NaV1.5 channel. Defaults to False.
        predict_Cav (bool, optional): Whether to predict CaV1.2 channel. Defaults to False.
        device (str, optional): The device to use for evaluation. Must be one of ['cpu', 'gpu']. Defaults to 'gpu'.
        herg_data (str, optional): The file path to the hERG dataset. Defaults to 'prepared_cardiac_datasets/test_hERG.h5'.
        nav_data (str, optional): The file path to the NaV1.5 dataset. Defaults to 'prepared_cardiac_datasets/test_Nav.h5'.
        cav_data (str, optional): The file path to the CaV1.2 dataset. Defaults to 'prepared_cardiac_datasets/test_Cav.h5'.
        herg_regression_params (str, optional): The file path to the hERG regression model parameters. Defaults to 'discriminative_model_parameters/hERG_Regression_parameters.pt'.
        nav_regression_params (str, optional): The file path to the NaV1.5 regression model parameters. Defaults to 'discriminative_model_parameters/Nav_Regression_parameters.pt'.
        cav_regression_params (str, optional): The file path to the CaV1.2 regression model parameters. Defaults to 'discriminative_model_parameters/Cav_Regression_parameters.pt'.
        herg_classification_params (str, optional): The file path to the hERG classification model parameters. Defaults to 'discriminative_model_parameters/hERG_Classification_parameters.pt'.
        nav_classification_params (str, optional): The file path to the NaV1.5 classification model parameters. Defaults to 'discriminative_model_parameters/Nav_Classification_parameters.pt'.
        cav_classification_params (str, optional): The file path to the CaV1.2 classification model parameters. Defaults to 'discriminative_model_parameters/Cav_Classification_parameters.pt'.
        save_dir (str, optional): The directory to save the evaluation results. Defaults to 'discriminative_results/'.

    Returns:
        pandas.DataFrame: The classification metrics if prediction_type is 'classification'.
    """

    os.makedirs(save_dir, exist_ok=True)

    assert device in [
        "cpu",
        "gpu",
    ], "Invalid device value. Must be one of ['cpu', 'gpu']"
    assert prediction_type in [
        "regression",
        "classification",
    ], "Invalid prediction type. Must be one of ['regression', 'classification']"
    assert (
        predict_hERG or predict_Nav or predict_Cav
    ), "At least one channel must be selected for evaluation."

    device = torch.device(
        "cuda:0" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
    )
    classification = True if prediction_type == "classification" else False

    # Load the discriminative models
    herg_regression_model = (
        load_discriminative_model(herg_regression_params, device)
        if herg_regression_params
        else None
    )
    cav_regression_model = (
        load_discriminative_model(cav_regression_params, device)
        if cav_regression_params
        else None
    )
    nav_regression_model = (
        load_discriminative_model(nav_regression_params, device)
        if nav_regression_params
        else None
    )
    herg_classification_model = (
        load_discriminative_model(herg_classification_params, device)
        if herg_classification_params
        else None
    )
    cav_classification_model = (
        load_discriminative_model(cav_classification_params, device)
        if cav_classification_params
        else None
    )
    nav_classification_model = (
        load_discriminative_model(nav_classification_params, device)
        if nav_classification_params
        else None
    )

    # Load the datasets
    herg_dataset = (
        Discriminator_Dataset(herg_data, classification) if predict_hERG else None
    )
    nav_dataset = (
        Discriminator_Dataset(nav_data, classification) if predict_Nav else None
    )
    cav_dataset = (
        Discriminator_Dataset(cav_data, classification) if predict_Cav else None
    )

    # Create the dataloaders
    herg_dataloader = (
        DataListLoader(herg_dataset, batch_size=32, shuffle=False, drop_last=False)
        if predict_hERG
        else None
    )
    nav_dataloader = (
        DataListLoader(nav_dataset, batch_size=32, shuffle=False, drop_last=False)
        if predict_Nav
        else None
    )
    cav_dataloader = (
        DataListLoader(cav_dataset, batch_size=32, shuffle=False, drop_last=False)
        if predict_Cav
        else None
    )

    results = {}

    # Perform inference with the discriminative models
    if predict_hERG:
        if prediction_type == "classification":
            herg_smiles, herg_predictions, herg_labels = predict_for_channel(
                herg_dataloader,
                prediction_type,
                herg_classification_model,
                "hERG",
                device,
                labels=True,
            )
        else:
            herg_smiles, herg_predictions, herg_labels = predict_for_channel(
                herg_dataloader,
                prediction_type,
                herg_regression_model,
                "hERG",
                device,
                labels=True,
            )
        results["hERG"] = {
            "smiles": herg_smiles,
            "predictions": herg_predictions,
            "labels": herg_labels,
        }

    if predict_Nav:
        if prediction_type == "classification":
            nav_smiles, nav_predictions, nav_labels = predict_for_channel(
                nav_dataloader,
                prediction_type,
                nav_classification_model,
                "NaV1.5",
                device,
                labels=True,
            )
        else:
            nav_smiles, nav_predictions, nav_labels = predict_for_channel(
                nav_dataloader,
                prediction_type,
                nav_regression_model,
                "NaV1.5",
                device,
                labels=True,
            )
        results["NaV1.5"] = {
            "smiles": nav_smiles,
            "predictions": nav_predictions,
            "labels": nav_labels,
        }

    if predict_Cav:
        if prediction_type == "classification":
            cav_smiles, cav_predictions, cav_labels = predict_for_channel(
                cav_dataloader,
                prediction_type,
                cav_classification_model,
                "CaV1.2",
                device,
                labels=True,
            )
        else:
            cav_smiles, cav_predictions, cav_labels = predict_for_channel(
                cav_dataloader,
                prediction_type,
                cav_regression_model,
                "CaV1.2",
                device,
                labels=True,
            )
        results["CaV1.2"] = {
            "smiles": cav_smiles,
            "predictions": cav_predictions,
            "labels": cav_labels,
        }

    # Save the results
    channel_results = []
    if predict_hERG:
        channel_results.append(("hERG", results["hERG"]))
    if predict_Nav:
        channel_results.append(("NaV1.5", results["NaV1.5"]))
    if predict_Cav:
        channel_results.append(("CaV1.2", results["CaV1.2"]))

    num_channels = len(channel_results)

    if prediction_type == "regression":

        # Create a scatter plot for each channel
        fig, axs = plt.subplots(1, num_channels, figsize=(6 * num_channels, 5))
        fig.suptitle("Discriminator Model Evaluation", fontsize=20, y=1.05)

        if num_channels == 1:
            axs = [axs]

        for i, (channel_name, channel_data) in enumerate(channel_results):
            axs[i].scatter(channel_data["predictions"], channel_data["labels"], s=3)
            axs[i].set_title(channel_name, fontsize=18, weight="bold")
            if i == 0:
                axs[i].set_ylabel("Actual pIC50", fontsize=16)
            if i == num_channels // 2:
                axs[i].set_xlabel("Predicted pIC50", fontsize=16)
            x = np.linspace(0, 12, 100)
            axs[i].plot(x, x, color="black")
            axs[i].set_xlim([0, 12])
            axs[i].set_ylim([0, 12])

            predictions_np = np.array(channel_data["predictions"])
            labels_np = torch.cat(channel_data["labels"]).cpu().detach().numpy()
            r = pearsonr(predictions_np, labels_np)[0]
            axs[i].text(2, 10, f"r = {r:.2f}", fontsize=16)
            axs[i].tick_params(axis="both", which="major", labelsize=14)

        plt.savefig(os.path.join(save_dir, "regression_evaluation.png"))
        print(
            f"Scatter plot saved to {os.path.join(save_dir, 'regression_evaluation.png')} ✅"
        )
        plt.show()

    elif prediction_type == "classification":

        # Create a DataFrame containing the classification metrics
        classification_results = []
        for channel_name, channel_data in channel_results:
            AC, SN, SP, F1, CCR, MCC = calculate_classification_metrics(
                channel_data["predictions"], channel_data["labels"]
            )
            classification_results.append(
                {
                    "Channel": channel_name,
                    "Accuracy (AC)": float(AC),
                    "Sensitivity (SN)": float(SN),
                    "Specificity (SP)": float(SP),
                    "F1-score (F1)": float(F1),
                    "Correct Classification Rate (CCR)": float(CCR),
                    "Matthew’s Correlation Coefficient (MCC)": float(MCC),
                }
            )

        df_classification_results = pd.DataFrame(classification_results)
        df_classification_results.to_csv(
            os.path.join(save_dir, "classification_evaluation.csv"), index=False
        )

        print(
            "Classification metrics saved to 'classification_evaluation.csv' See DataFrame below: ⬇️"
        )

        return df_classification_results


# Define a function to screen FDA compounds for cardiac ion channel activity
def screen_FDA_compounds(
    input_data="data/raw_cardiac_datasets/FDA_compounds.txt",
    prediction_type="regression",
    predict_hERG=True,
    predict_Nav=True,
    predict_Cav=True,
    rank_by="hERG",
    device="gpu",
    herg_regression_params="model_parameters/discriminative_model_parameters/hERG_Regression_parameters.pt",
    nav_regression_params="model_parameters/discriminative_model_parameters/Nav_Regression_parameters.pt",
    cav_regression_params="model_parameters/discriminative_model_parameters/Cav_Regression_parameters.pt",
    herg_classification_params="model_parameters/discriminative_model_parameters/hERG_Classification_parameters.pt",
    nav_classification_params="model_parameters/discriminative_model_parameters/Nav_Classification_parameters.pt",
    cav_classification_params="model_parameters/discriminative_model_parameters/Cav_Classification_parameters.pt",
    bidirectional_transformer_params="model_parameters/transformer_model_parameters/Bidirectional_Transformer_parameters.pt",
    transformer_training_data="data/prepared_transformer_datasets/prepared_transformer_data.csv",
    save_path="results/discriminative_results/FDA_compound_predictions.json",
    save=True,
):
    """
    Screen FDA compounds for cardiac ion channel activity.

    Args:
        input_data (str): Path to the input data file containing SMILES strings of FDA compounds.
        prediction_type (str): Type of prediction to perform. Can be 'regression' or 'classification'.
        predict_hERG (bool): Whether to predict hERG activity.
        predict_Nav (bool): Whether to predict Nav activity.
        predict_Cav (bool): Whether to predict Cav activity.
        rank_by (str): The ion channel to rank the results by. Can be 'hERG', 'NaV1.5', or 'CaV1.2'.
        device (str): The device to use for prediction. Can be 'cpu' or 'gpu'.
        herg_regression_params (str): Path to the hERG regression model parameters file.
        nav_regression_params (str): Path to the Nav regression model parameters file.
        cav_regression_params (str): Path to the Cav regression model parameters file.
        herg_classification_params (str): Path to the hERG classification model parameters file.
        nav_classification_params (str): Path to the Nav classification model parameters file.
        cav_classification_params (str): Path to the Cav classification model parameters file.
        bidirectional_transformer_params (str): Path to the bidirectional transformer model parameters file.
        transformer_training_data (str): Path to the prepared transformer training data file.
        save_path (str): Path to save the results file.
        save (bool): Whether to save the results to a file.

    Returns:
        pandas.DataFrame: DataFrame containing the predicted ion channel activity scores for each compound.

    Raises:
        AssertionError: If `rank_by` is not one of 'hERG', 'NaV1.5', or 'CaV1.2'.
    """

    assert rank_by in [
        "hERG",
        "NaV1.5",
        "CaV1.2",
    ], "rank_by_channel must be one of 'hERG', 'NaV1.5', or 'CaV1.2'."

    # Retrieve the input SMILES strings
    input_smiles_list = pd.read_csv(input_data)["SMILES"].tolist()

    # Perform prediction of cardiac ion channel activity for each compound
    results = predict_cardiac_ion_channel_activity(
        input_data=input_smiles_list,
        prediction_type=prediction_type,
        predict_hERG=predict_hERG,
        predict_Nav=predict_Nav,
        predict_Cav=predict_Cav,
        device=device,
        bidirectional_transformer_params=bidirectional_transformer_params,
        transformer_training_data=transformer_training_data,
        herg_regression_params=herg_regression_params,
        nav_regression_params=nav_regression_params,
        cav_regression_params=cav_regression_params,
        herg_classification_params=herg_classification_params,
        nav_classification_params=nav_classification_params,
        cav_classification_params=cav_classification_params,
        save_path=save_path,
    )

    # Initialize an empty DataFrame
    df_results = pd.DataFrame(results["smiles"].unique(), columns=["SMILES"])

    for channel in ["hERG", "NaV1.5", "CaV1.2"]:

        # Create a dictionary mapping 'SMILES' to the channel's predictions
        smiles_scores = dict(zip(results["smiles"], results[channel]))

        # Map predictions to the corresponding 'SMILES' in df_results,
        df_results[channel] = df_results["SMILES"].map(smiles_scores).fillna(0)

    # Sort the results based on a specific channel
    df_results = df_results.sort_values(by=rank_by, ascending=False)

    if save:

        # Save the results
        df_results.to_csv(save_path.replace(".json", ".csv"), index=False)
        print(f"Results saved to {save_path.replace('.json', '.csv')}")

    return df_results


# Define a function to plot the ROC curve for each cardiac ion channel
def plot_roc_curve_for_each_cardiac_ion_channel(
    parameters_directory="model_parameters/discriminative_model_parameters",
    data_directory="data/prepared_cardiac_datasets",
    device="gpu",
    save_dir="results/discriminative_results",
):
    """
    Plots the Receiver Operating Characteristic (ROC) curve for each cardiac ion channel.

    Args:
        parameters_directory (str): The directory path where the model parameters are stored. Default is 'discriminative_model_parameters'.
        data_directory (str): The directory path where the cardiac datasets are stored. Default is 'prepared_cardiac_datasets'.
        device (str): The device to run the model on. Can be 'cpu' or 'gpu'. Default is 'cpu'.
        save_dir (str): The directory path to save the generated ROC curve plot. Default is 'discriminative_results'.

    Returns:
        None
    """

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(
        "cuda:0" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
    )
    model_files = [
        f
        for f in os.listdir(parameters_directory)
        if f.endswith(".pt") and "Classification" in f
    ]

    all_results = []

    for model_file in model_files:
        # Load the appropriate dataset
        model_path = os.path.join(parameters_directory, model_file)
        if "hERG" in model_file:
            name = "hERG"
            for file in os.listdir(data_directory):
                if "hERG" in file and "test" in file:
                    herg_data = os.path.join(data_directory, file)
            dataset = Discriminator_Dataset(herg_data, classification=True)
        elif "Nav" in model_file:
            name = "NaV1.5"
            for file in os.listdir(data_directory):
                if "Nav" in file and "test" in file:
                    nav_data = os.path.join(data_directory, file)
            dataset = Discriminator_Dataset(nav_data, classification=True)
        elif "Cav" in model_file:
            name = "CaV1.2"
            for file in os.listdir(data_directory):
                if "Cav" in file and "test" in file:
                    cav_data = os.path.join(data_directory, file)
            dataset = Discriminator_Dataset(cav_data, classification=True)

        dataloader = DataListLoader(
            dataset, batch_size=32, shuffle=False, drop_last=False
        )

        # Build the model
        model = Discriminator_Model(device=device).float()
        model_train_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(model_train_dict["model_state_dict"], strict=False)
        model.to(device)
        model.eval()

        # Perform inference with the discriminative models
        _, probabilities, labels = predict_for_channel(
            dataloader, "classification", model, name, device, labels=True, logits=True
        )

        # Calculate the ROC curve and AUC
        fpr, tpr, _ = roc_curve(labels, probabilities)
        roc_auc = auc(fpr, tpr)

        all_results.append((name, {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc}))
        all_results.sort(key=lambda x: x[1]["roc_auc"], reverse=True)

    # Plot and save the ROC curve for each cardiac ion channel
    plt.figure(figsize=(15, 9))
    colors = ["#E57373", "#81C784", "#9575CD"]
    for i, (name, results) in enumerate(all_results):
        plt.plot(
            results["fpr"],
            results["tpr"],
            color=colors[i],
            lw=2,
            label=f'{name} (AUC = {results["roc_auc"]:.2f})',
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
    plt.legend(loc="lower right", fontsize=20, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=300)
    plt.show()
