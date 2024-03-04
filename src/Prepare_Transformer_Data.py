# Import necessary libraries
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from tqdm import tqdm

from src.utils import get_scaffold

# Enable tqdm for pandas
tqdm.pandas()


# Define a function to prepare the transformer data
def prepare_transformer_data(input_file, output_file):
    """
    Preprocesses the input data file and calculates various molecular properties for each SMILES string.

    Args:
        input_file (str): Path to the input data file.
        output_file (str): Path to the output file where the preprocessed data will be saved.

    Returns:
        None
    """

    # Load the input data and remove duplicate SMILES strings
    df = pd.read_csv(input_file)
    df = df.drop_duplicates(subset=["SMILES"])

    # Calculate ten ADMET properties for each SMILES string
    properties = [
        (
            "MW",
            lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)),
            "Calculating Molecular Weight",
        ),
        (
            "HBA",
            lambda x: rdMolDescriptors.CalcNumHBA(Chem.MolFromSmiles(x)),
            "Calculating HBA",
        ),
        (
            "HBD",
            lambda x: rdMolDescriptors.CalcNumHBD(Chem.MolFromSmiles(x)),
            "Calculating HBD",
        ),
        (
            "nRot",
            lambda x: Descriptors.NumRotatableBonds(Chem.MolFromSmiles(x)),
            "Calculating Rotatable Bonds",
        ),
        (
            "nRing",
            lambda x: rdMolDescriptors.CalcNumRings(Chem.MolFromSmiles(x)),
            "Calculating Rings",
        ),
        (
            "nHet",
            lambda x: rdMolDescriptors.CalcNumHeteroatoms(Chem.MolFromSmiles(x)),
            "Calculating Heteroatoms",
        ),
        (
            "fChar",
            lambda x: Chem.GetFormalCharge(Chem.MolFromSmiles(x)),
            "Calculating Formal Charge",
        ),
        (
            "TPSA", 
            lambda x: Descriptors.TPSA(Chem.MolFromSmiles(x)), 
            "Calculating TPSA"
        ),
        (
            "LogP", 
            lambda x: Crippen.MolLogP(Chem.MolFromSmiles(x)), 
            "Calculating LogP"
        ),
        (
            "StereoCenters",
            lambda x: len(
                Chem.FindMolChiralCenters(Chem.MolFromSmiles(x), includeUnassigned=True)
            ),
            "Calculating Stereocenters",
        ),
        ("scaffold", get_scaffold, "Calculating Scaffolds"),
    ]

    for prop, func, desc in properties:
        print(desc)
        df[prop] = df["SMILES"].progress_apply(func)

    # Save the preprocessed data to a CSV file
    df.to_csv(output_file, index=False)
