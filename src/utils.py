# Import necessary libraries
import base64
import math
import numpy as np
from openbabel import pybel
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import pearsonr
import torch
from torch_geometric.data import Batch
from tqdm import tqdm


# Define a class to featurize molecules
class Featurizer(object):
    """
    A class that provides methods for featurizing molecules.

    Attributes:
        atom_classes (list): A list of tuples representing the atomic number and symbol for each atom class.
        ATOM_CODES (dict): A dictionary mapping atomic numbers to their corresponding codes.
        NAMED_PROPS (list): A list of named properties.
        SMARTS (list): A list of SMARTS patterns.
        __PATTERNS (list): A list of pybel.Smarts objects created from the SMARTS patterns.
        FEATURE_NAMES (list): A list of feature names.

    Methods:
        encode_num(atomic_num): Encodes the atomic number into a one-hot vector.
        find_smarts(molecule): Finds the presence of SMARTS patterns in a molecule.
        get_features(molecule): Generates the features for a molecule.

    """

    # Define the class constructor
    def __init__(self):

        # Define the atom classes
        self.atom_classes = [(6, "C"), (7, "N"), (8, "O"), (15, "P"), (16, "S")]
        self.ATOM_CODES = {
            atom: code for code, (atom, _) in enumerate(self.atom_classes)
        }

        # Define the named properties and SMARTS patterns
        self.NAMED_PROPS = ["heavydegree", "heterodegree", "partialcharge"]
        self.SMARTS = [
            "[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]",
            "[a]",
            "[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]",
            "[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]",
            "[r]",
        ]
        self.__PATTERNS = [pybel.Smarts(smarts) for smarts in self.SMARTS]

        # Define complete feature list
        self.FEATURE_NAMES = (
            ["C", "N", "O", "P", "S"]
            + self.NAMED_PROPS
            + ["molcode"]
            + ["hydrophobic", "aromatic", "acceptor", "donor", "ring"]
        )

    # Method to encode the atomic number into a one-hot vector
    def encode_num(self, atomic_num):
        encoding = np.zeros(len(self.atom_classes))
        if atomic_num in self.ATOM_CODES:
            encoding[self.ATOM_CODES[atomic_num]] = 1.0
        return encoding

    # Method to find the presence of SMARTS patterns in a molecule
    def find_smarts(self, molecule):
        features = np.zeros((len(molecule.atoms), len(self.__PATTERNS)))
        for pattern_id, pattern in enumerate(self.__PATTERNS):
            atoms_with_prop = (
                np.array(
                    [atom_idx for atom_idx in pattern.findall(molecule)], dtype=int
                )
                - 1
            )
            features[atoms_with_prop, pattern_id] = 1.0
        return features

    # Method to calculate the features for a molecule
    def get_features(self, molecule):
        features, heavy_atoms = [], []
        for i, atom in enumerate(molecule):
            if atom.atomicnum > 1:

                # Store the index of heavy atoms
                heavy_atoms.append(i)

                # Calculate features for the heavy atom
                features.append(
                    np.concatenate(
                        (
                            self.encode_num(atom.atomicnum),
                            [atom.__getattribute__(prop) for prop in self.NAMED_PROPS],
                            [atom.atomicmass],
                        )
                    )
                )

        # Format and return the features
        features = np.array(features, dtype=np.float32)
        features = np.hstack([features, self.find_smarts(molecule)[heavy_atoms]])
        return features


# Define a function to encode a SMILES string
def encode_smiles(smiles: str) -> str:
    """
    Encodes a SMILES string using base64 encoding.

    Args:
        smiles (str): The SMILES string to be encoded.

    Returns:
        str: The encoded SMILES string.

    """
    return base64.urlsafe_b64encode(smiles.encode()).decode()


# Define a function to decode an encoded SMILES string
def decode_smiles(encoded_smiles: str) -> str:
    """
    Decodes a base64 encoded string representing SMILES notation and returns the decoded string.

    Args:
        encoded_smiles (str): The base64 encoded string representing SMILES notation.

    Returns:
        str: The decoded SMILES string.

    """
    return base64.urlsafe_b64decode(encoded_smiles.encode()).decode()


# Define a function to calculate the graph features for a molecule
def get_graph_features(smiles: str):
    """
    Extracts graph features from a given SMILES string.

    Parameters:
        smiles (str): The SMILES string representing the molecule.

    Returns:
        feats (list): A list of graph features extracted from the molecule.
    """

    # Create a featurizer object
    featurizer = Featurizer()

    # Get the graph features for the molecule
    mol = pybel.readstring("smi", smiles)
    feats = featurizer.get_features(mol)

    return feats


# Define a function to calculate the fingerprint features for a molecule
def get_fingerprint_features(smiles: str) -> np.ndarray:
    """
    Generates fingerprint features for a given SMILES string.

    Parameters:
        smiles (str): The SMILES string representing a molecule.

    Returns:
        np.ndarray: An array containing the fingerprint features.
    """
    # Get the fingerprint for the molecule
    mol = Chem.MolFromSmiles(smiles)
    ecfp2_fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)

    return np.array(ecfp2_fingerprint)


# Define a function to calculate the scaffold for a molecule
def get_scaffold(smiles):
    """
    Get the scaffold for a given SMILES string.

    Parameters:
    smiles (str): The SMILES string representing the molecule.

    Returns:
    str: The SMILES string representing the scaffold.
    """

    # Get the scaffold for the molecule
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)

    # Convert the scaffold to a SMILES string
    scaffold_smiles = Chem.MolToSmiles(scaffold)

    return scaffold_smiles


# Define a function to calculate ten ADMET properties for a molecule
def get_admet_properties(smiles):
    """
    Calculates various ADMET properties for a given molecule.

    Parameters:
    - smiles (str): The SMILES representation of the molecule.

    Returns:
    - List[float]: A list of calculated ADMET properties for the molecule.
    """

    mol = Chem.MolFromSmiles(smiles)

    # Calculate the properties
    properties = [
        ("MW", Descriptors.MolWt(mol)),
        ("HBA", rdMolDescriptors.CalcNumHBA(mol)),
        ("HBD", rdMolDescriptors.CalcNumHBD(mol)),
        ("nRot", Descriptors.NumRotatableBonds(mol)),
        ("nRing", rdMolDescriptors.CalcNumRings(mol)),
        ("nHet", rdMolDescriptors.CalcNumHeteroatoms(mol)),
        ("fChar", Chem.GetFormalCharge(mol)),
        ("TPSA", Descriptors.TPSA(mol)),
        ("LogP", Crippen.MolLogP(mol)),
        ("StereoCenters", len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))),
    ]

    return [value for _, value in properties]


# Define a function to get the most similar scaffold in a provided dataset
def get_most_similar_scaffold(input_scaffold_smiles, search_data):
    """
    Finds the most similar scaffold to the input scaffold based on Tanimoto similarity.

    Parameters:
    input_scaffold_smiles (str): The SMILES representation of the input scaffold.
    search_data (str): The filepath to the CSV file containing the search data.

    Returns:
    str: The SMILES representation of the most similar scaffold.
    """

    # Load the dataset
    search_data_df = pd.read_csv(search_data)

    # Get the fingerprint for the input scaffold
    mol = Chem.MolFromSmiles(input_scaffold_smiles)
    input_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)

    # Initialize variables to store the maximum similarity and the most similar scaffold
    max_similarity = 0
    most_similar_scaffold = None

    for _, row in search_data_df.iterrows():
        scaffold_smiles = row["scaffold"]

        # Get the fingerprint for the scaffold
        scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
        scaffold_fp = AllChem.GetMorganFingerprintAsBitVect(
            scaffold_mol, radius=2, nBits=1024
        )

        # Calculate the Tanimoto similarity between the input scaffold and the current scaffold
        similarity = DataStructs.TanimotoSimilarity(input_fp, scaffold_fp)

        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_scaffold = scaffold_smiles

    return most_similar_scaffold


# Define a function to calculate L1 regularization
def l1_regularization(model):
    """
    Calculate the L1 regularization loss for a given model.

    Parameters:
    model (torch.nn.Module): The model for which to calculate the L1 regularization loss.

    Returns:
    torch.Tensor: The L1 regularization loss.
    """

    l1_reg = 0
    for param in model.parameters():
        # Calculate the L1 norm for each parameter
        l1_reg += torch.norm(param, 1)

    return l1_reg


# Define a function to validate the discriminative models
def validate(model, validation_dataloader, classification=True, epoch=0, device="gpu"):
    """
    Perform validation on the given model using the provided validation dataloader.

    Args:
        model (torch.nn.Module): The model to be validated.
        validation_dataloader (torch.utils.data.DataLoader): The dataloader containing the validation data.
        classification (bool, optional): Whether the task is classification or regression. Defaults to True.
        epoch (int, optional): The current epoch number. Defaults to 0.
        device (str, optional): The device to run the validation on. Defaults to 'cpu'.

    Returns:
        dict: A dictionary containing the loss and score of the validation.
    """

    model.eval()

    total_loss = 0
    total_score = 0
    total_samples = 0

    loss_function = (
        torch.nn.BCEWithLogitsLoss() if classification else torch.nn.MSELoss()
    )

    with torch.no_grad():
        with tqdm(
            validation_dataloader, desc=f"Epoch {epoch} Validation", leave=True
        ) as t:
            for batch in t:

                # Retrieve the graph data, fingerprint, transformer feature vector, and labels
                graph_data_list, fingerprint_list, transformer_vector_list, labels = (
                    [],
                    [],
                    [],
                    [],
                )
                for data in batch:
                    graph_data_list.append(data)
                    fingerprint_list.append(data.fingerprint)
                    transformer_vector_list.append(
                        torch.tensor(data.transformer_vector, dtype=torch.float)
                    )
                    labels.append(data.label)

                # Convert the data to the appropriate format and move it to the device
                batch_graph_data = Batch.from_data_list(graph_data_list).to(device)
                batch_fingerprint = torch.stack(fingerprint_list).to(device)
                batch_transformer_vector = torch.stack(transformer_vector_list).to(
                    device
                )

                # Perform a forward pass
                y_pred = model(
                    graph_data=batch_graph_data,
                    fingerprint=batch_fingerprint,
                    transformer_features=batch_transformer_vector,
                ).to(device)

                # Calculate the loss
                y_true = torch.tensor(labels, dtype=torch.float).to(device)
                loss = loss_function(y_pred.squeeze(1), y_true).to(device)

                total_loss += loss.item() * y_true.size(0)
                total_samples += y_true.size(0)

                if classification:

                    # Calculate the accuracy
                    probabilities = torch.sigmoid(y_pred)
                    predictions = (probabilities > 0.5).float()
                    predictions = predictions.squeeze(1)
                    accuracy = (predictions == y_true).float().mean().item()
                    total_score += accuracy * y_true.size(0)

                    t.set_postfix(
                        Loss=total_loss / total_samples,
                        Accuracy=total_score / total_samples,
                    )

                elif not classification:

                    # Calculate the Pearson correlation coefficient
                    correlation = pearsonr(
                        y_pred.squeeze(1).cpu().numpy(), y_true.cpu().numpy()
                    )[0]
                    total_score += correlation * y_true.size(0)
                    t.set_postfix(
                        Loss=total_loss / total_samples,
                        Correlation=total_score / total_samples,
                    )

        total_loss /= total_samples
        total_score /= total_samples

    model.train()

    return {"loss": total_loss, "score": total_score}


# Define a function to filter generated molecules based on their cardiac ion channel activity
def filter_cardiac_ion_channel_activity(
    activity, classification_model, regression_model, data, device
):
    """
    Filters cardiac ion channel activity based on the given activity type and models.

    Parameters:
    - activity (str or tuple): The type of activity to filter for. If it's a string, it can be either 'blockers' or 'non-blockers'.
      If it's a tuple, it should contain a range of activity values.
    - classification_model: The classification model used to predict activity for 'blockers' and 'non-blockers'.
    - regression_model: The regression model used to predict activity for the given range.
    - data: The data object containing the fingerprint and transformer features.
    - device: The device to run the models on.

    Returns:
    - meets_criteria (bool): Indicates whether the activity meets the specified criteria.
    - y_pred (float or None): The predicted activity value. None if the activity type is not applicable.
    """

    fingerprint, transformer_features = data.fingerprint, data.transformer_vector

    if isinstance(activity, str):
        # Perform classification
        y_pred = classification_model(data, fingerprint, transformer_features).to(
            device
        )
        meets_criteria = (
            (y_pred.item() >= 0.5) if activity == "blockers" else (y_pred.item() < 0.5)
        )

    elif isinstance(activity, tuple):
        # Perform regression
        y_pred = regression_model(data, fingerprint, transformer_features).to(device)
        meets_criteria = activity[0] < y_pred.item() < activity[1]

    else:
        # If the activity type is not applicable, set the meets_criteria flag to True
        meets_criteria = True
        y_pred = None

    return meets_criteria, None if y_pred is None else y_pred.item()


# Define a function to predict the activity against a given channel
def predict_for_channel(
    dataloader, prediction_type, model, channel_desc, device, labels=False, logits=False
):
    """
    Predicts the output for a given channel using the provided dataloader and model.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader containing the input data.
        prediction_type (str): The type of prediction to perform ('classification' or 'regression').
        model: The model used for prediction.
        channel_desc (str): The description of the channel.
        device: The device to perform the prediction on.
        labels (bool, optional): Whether to include labels in the output. Defaults to False.
        logits (bool, optional): Whether to return logits instead of probabilities for classification. Defaults to False.

    Returns:
        tuple: A tuple containing the predicted smiles, predictions, and labels (if labels=True).
    """

    smiles_list, predictions_list, labels_list = [], [], []

    with torch.no_grad():
        with tqdm(
            dataloader,
            desc=f"{channel_desc} Prediction ({prediction_type.capitalize()})",
            leave=True,
        ) as t:
            for batch in t:

                # Retrieve the graph data, fingerprint, transformer feature vector, and labels
                (
                    graph_data_list,
                    fingerprint_list,
                    transformer_vector_list,
                    batch_smiles,
                    batch_labels,
                ) = ([], [], [], [], [])

                for data in batch:
                    graph_data_list.append(data)
                    fingerprint_list.append(data.fingerprint)

                    if isinstance(data.transformer_vector, torch.Tensor):
                        transformer_vector_list.append(
                            data.transformer_vector.to(torch.float)
                        )

                    else:
                        transformer_vector_list.append(
                            torch.tensor(data.transformer_vector, dtype=torch.float)
                        )

                    batch_smiles.append(data.smiles)

                    if labels:
                        batch_labels.append(data.label)

                # Convert the data to the appropriate format and move it to the device
                batch_graph_data = Batch.from_data_list(graph_data_list).to(device)
                batch_fingerprint = torch.stack(fingerprint_list).to(device)
                batch_transformer_vector = torch.stack(transformer_vector_list).to(
                    device
                )

                # Perform a forward pass
                y_pred = model(
                    graph_data=batch_graph_data,
                    fingerprint=batch_fingerprint,
                    transformer_features=batch_transformer_vector,
                ).to(device)

                if prediction_type == "classification":
                    probabilities = torch.sigmoid(y_pred)

                    if logits:
                        predictions = probabilities

                    else:
                        predictions = (probabilities > 0.5).float().squeeze(1)

                else:
                    predictions = y_pred.squeeze(1)

                smiles_list += batch_smiles
                predictions_list += predictions.cpu().numpy().tolist()

                if labels:
                    labels_list += batch_labels

    if labels:
        return smiles_list, predictions_list, labels_list

    return smiles_list, predictions_list


# Define a function to calculate classification metrics
def calculate_classification_metrics(predictions, labels):
    """
    Calculates various classification metrics based on the predictions and labels.

    Args:
        predictions (list): A list of predicted values (0 or 1).
        labels (list): A list of true labels (0 or 1).

    Returns:
        tuple: A tuple containing the following classification metrics:
            - AC (float): Accuracy.
            - SN (float): Sensitivity (True Positive Rate).
            - SP (float): Specificity (True Negative Rate).
            - PR (float): Precision.
            - F1 (float): F1 Score.
            - CCR (float): Correct Classification Rate.
            - MCC (float): Matthews Correlation Coefficient.
    """

    # Calculate the true positives, false positives, true negatives, and false negatives
    true_positives = sum(
        (pred == 1) and (label == 1) for pred, label in zip(predictions, labels)
    )
    false_positives = sum(
        (pred == 1) and (label == 0) for pred, label in zip(predictions, labels)
    )
    true_negatives = sum(
        (pred == 0) and (label == 0) for pred, label in zip(predictions, labels)
    )
    false_negatives = sum(
        (pred == 0) and (label == 1) for pred, label in zip(predictions, labels)
    )

    # Calculate accuracy (AC), sensitivity (SN), specificity (SP), precision (PR), 
    # F1 score, correct classification rate (CCR), and Matthews correlation coefficient (MCC)
    AC = (true_positives + true_negatives) / (
        true_positives + true_negatives + false_positives + false_negatives
    )
    SN = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    PR = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    SP = (
        true_negatives / (true_negatives + false_positives)
        if (true_negatives + false_positives) > 0
        else 0
    )
    F1 = 2 * ((PR * SN) / (PR + SN)) if (PR + SN) > 0 else 0
    CCR = (SN + SP) / 2
    MCC_numerator = (true_positives * true_negatives) - (
        false_positives * false_negatives
    )
    MCC_denominator = math.sqrt(
        (true_positives + false_positives)
        * (true_positives + false_negatives)
        * (true_negatives + false_positives)
        * (true_negatives + false_negatives)
    )
    MCC = MCC_numerator / MCC_denominator if MCC_denominator > 0 else 0

    return AC, SN, SP, F1, CCR, MCC
