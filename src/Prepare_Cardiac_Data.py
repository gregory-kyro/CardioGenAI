# Import necessary libraries
import h5py
import os
import pandas as pd
from tqdm import tqdm

from src.Transformer import Transformer_Feature_Extractor
from src.utils import (
    encode_smiles,
    get_fingerprint_features,
    get_graph_features,
)


# Define a function to prepare the cardiac dataset
def prepare_cardiac_dataset(
    data_csv: str,
    h5_save_path: str,
    transformer_feature_extractor,
    dataset_type=None,
    tqdm_desc="Processing molecules",
    verbose=False,
):
    """
    Prepare the cardiac dataset by processing molecules and saving them in an HDF5 file.

    Args:
        data_csv (str): Path to the CSV file containing the dataset.
        h5_save_path (str): Path to save the HDF5 file.
        transformer_feature_extractor: The feature extractor for the transformer model.
        dataset_type (str, optional): Type of dataset to prepare. Defaults to None.
        tqdm_desc (str, optional): Description for the tqdm progress bar. Defaults to 'Processing molecules'.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """

    # Load and process the dataset
    df = pd.read_csv(data_csv)
    df = df[df["USED_AS"] == dataset_type] if dataset_type != "Test" else df
    df = df[df["SMILES"].apply(len) <= 133]

    # Write the processed data to an HDF5 file
    with h5py.File(h5_save_path, "w") as f:
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=tqdm_desc):

            # Retrieve the SMILES string
            smiles = row["SMILES"]

            # Calculate the graph features
            graph_feats = get_graph_features(smiles)

            # Calculate the fingerprint features
            fingerprint_feats = get_fingerprint_features(smiles)

            # Extract the transformer features
            try:
                transformer_feats = transformer_feature_extractor.extract_features(
                    smiles
                )[0, :]

                # Encode the SMILES string for acceptable HDF5 group name
                encoded_smiles = encode_smiles(smiles)

                # Save each molecule's features as a group in the HDF5 file
                mol_group = f.create_group(encoded_smiles)
                mol_group.create_dataset("graph", data=graph_feats, compression="gzip")
                mol_group.create_dataset(
                    "fingerprint", data=fingerprint_feats, compression="gzip"
                )
                mol_group.create_dataset("transformer_vector", data=transformer_feats)
                mol_group.attrs["label"] = row["pIC50"]

            # Skip molecules with tokens not present in the Transformer's vocabulary (very rare)
            except Exception:
                (
                    print(
                        f"SMILES {smiles} contains a token not present in the Transformer's vocabulary"
                    )
                    if verbose
                    else None
                )


# Define a function to prepare the cardiac datasets
def prepare_cardiac_datasets(input_dir: str, output_dir: str, verbose=False):
    """
    Prepare cardiac datasets by processing input files from the input directory and saving the processed files to the output directory.

    Args:
        input_dir (str): The directory path where the input files are located.
        output_dir (str): The directory path where the processed files will be saved.
        verbose (bool, optional): Whether to display verbose output. Defaults to False.
    """

    os.makedirs(output_dir, exist_ok=True)

    transformer_feature_extractor = Transformer_Feature_Extractor()

    # Process each file in the input directory
    for file_name in os.listdir(input_dir):

        if "train_val" in file_name:
            for dataset_type in ["Train", "Validation"]:
                input_file_path = os.path.join(input_dir, file_name)
                output_file_name = (
                    file_name.replace("raw_", "")
                    .replace("train_val", dataset_type.lower())
                    .replace(".csv", ".h5")
                )
                output_file_path = os.path.join(output_dir, output_file_name)
                tqdm_desc = f"Processing {dataset_type} file from {file_name}"
                prepare_cardiac_dataset(
                    input_file_path,
                    output_file_path,
                    transformer_feature_extractor,
                    dataset_type,
                    tqdm_desc,
                    verbose,
                )

        elif "test" in file_name:
            input_file_path = os.path.join(input_dir, file_name)
            output_file_name = file_name.replace("raw_", "").replace(".csv", ".h5")
            output_file_path = os.path.join(output_dir, output_file_name)
            tqdm_desc = f"Processing Test file from {file_name}"
            prepare_cardiac_dataset(
                input_file_path,
                output_file_path,
                transformer_feature_extractor,
                "Test",
                tqdm_desc,
                verbose,
            )
