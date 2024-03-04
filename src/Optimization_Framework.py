# Import necessary libraries
import numpy as np
import os
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
from sklearn.metrics import mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from tqdm import tqdm
import warnings

from src.Discriminator import (
    load_discriminative_model,
    prepare_data_for_discriminative_models,
)
from src.Transformer import (
    Transformer_Dataset,
    Transformer_Feature_Extractor,
    Transformer_Model,
)
from src.utils import (
    filter_cardiac_ion_channel_activity,
    get_admet_properties,
    get_most_similar_scaffold,
    get_scaffold,
)


# Define a class for generating optimized molecules
class Generation_Framework(object):
    """
    Class representing the generation framework for optimizing molecules.

    Args:
        autoregressive_transformer_params (str): Path to the autoregressive transformer model parameters.
        bidirectional_transformer_params (str): Path to the bidirectional transformer model parameters.
        transformer_training_data (str): Path to the transformer training data.
        herg_regression_params (str): Path to the hERG regression model parameters.
        cav_regression_params (str): Path to the CaV1.2 regression model parameters.
        nav_regression_params (str): Path to the NaV1.5 regression model parameters.
        herg_classification_params (str): Path to the hERG classification model parameters.
        cav_classification_params (str): Path to the CaV1.2 classification model parameters.
        nav_classification_params (str): Path to the NaV1.5 classification model parameters.
        device (str): Device to use for computation. Must be one of ['cpu', 'gpu'].

    Attributes:
        device (torch.device): Device to use for computation.
        transformer_training_data (str): Path to the transformer training data.
        dataset (Transformer_Dataset): Dataset for the autoregressive transformer model.
        model (Transformer_Model): Autoregressive transformer model.
        herg_regression_model (Discriminative_Model): hERG regression model.
        cav_regression_model (Discriminative_Model): CaV1.2 regression model.
        nav_regression_model (Discriminative_Model): NaV1.5 regression model.
        herg_classification_model (Discriminative_Model): hERG classification model.
        cav_classification_model (Discriminative_Model): CaV1.2 classification model.
        nav_classification_model (Discriminative_Model): NaV1.5 classification model.
        transformer_feature_extractor (Transformer_Feature_Extractor): Bidirectional transformer feature extractor.

    """

    def __init__(
        self,
        autoregressive_transformer_params=None,
        bidirectional_transformer_params=None,
        transformer_training_data=None,
        herg_regression_params=None,
        cav_regression_params=None,
        nav_regression_params=None,
        herg_classification_params=None,
        cav_classification_params=None,
        nav_classification_params=None,
        device="gpu",
    ):

        assert device in [
            "cpu",
            "gpu",
        ], "Invalid device value. Must be one of ['cpu', 'gpu']"
        self.device = torch.device(
            "cuda:0" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
        )

        print("Loading models... (This will take about 2 minutes)")

        # Load the transformer data
        self.transformer_training_data = transformer_training_data
        self.dataset = Transformer_Dataset(
            mode="Autoregressive", data_path=transformer_training_data
        )

        # Load the autoregressive transformer model
        self.model = Transformer_Model(
            mode="Autoregressive",
            vocab_size=len(self.dataset.vocab),
            block_size=self.dataset.block_size,
            admet_dim=10,
            num_scaffolds=self.dataset.num_scaffolds,
        ).to(self.device)
        checkpoint = torch.load(
            autoregressive_transformer_params, map_location=self.device
        )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()

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

        # Load the bidirectional transformer feature extractor
        self.transformer_feature_extractor = Transformer_Feature_Extractor(
            model_parameters=bidirectional_transformer_params,
            training_data=self.transformer_training_data,
            device=device,
        )

    # Method to generate optimized molecules
    def generate_smiles(
        self,
        input_smiles=None,
        num=100,
        herg_activity=None,
        cav_activity=None,
        nav_activity=None,
        save_path=None,
    ):

        for filter, string in zip(
            [herg_activity, cav_activity, nav_activity],
            ["herg_activity", "cav_activity", "nav_activity"],
        ):
            if filter is not None:
                assert filter in ["blocker", "non-blocker"] or isinstance(
                    filter, tuple
                ), f"Invalid {string} value"

        if os.path.dirname(save_path) and not os.path.exists(
            os.path.dirname(save_path)
        ):
            os.makedirs(os.path.dirname(save_path))

        # Get canonical input SMILES
        canonical_input_smiles = Chem.MolToSmiles(
            Chem.MolFromSmiles(input_smiles), canonical=True
        )

        # Prepare input data for discriminative models
        input_data = prepare_data_for_discriminative_models(
            canonical_input_smiles,
            self.transformer_feature_extractor,
            self.device,
            add_batch_dim=True,
        )

        # Predict activity against each of the cardiac ion channels
        _, herg_pred = filter_cardiac_ion_channel_activity(
            herg_activity,
            self.herg_classification_model,
            self.herg_regression_model,
            input_data,
            self.device,
        )
        _, cav_pred = filter_cardiac_ion_channel_activity(
            cav_activity,
            self.cav_classification_model,
            self.cav_regression_model,
            input_data,
            self.device,
        )
        _, nav_pred = filter_cardiac_ion_channel_activity(
            nav_activity,
            self.nav_classification_model,
            self.nav_regression_model,
            input_data,
            self.device,
        )

        # Store the SMILES string and corresponding ADMET properties
        input_admet_properties = get_admet_properties(canonical_input_smiles)
        admet_labels = [
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
        ]
        input_data_entry = {
            "SMILES": canonical_input_smiles,
            **dict(zip(admet_labels, input_admet_properties)),
        }

        # Store the predictions
        if herg_activity is not None:
            input_data_entry["hERG pIC50"] = herg_pred
        if cav_activity is not None:
            input_data_entry["CaV1.2 pIC50"] = cav_pred
        if nav_activity is not None:
            input_data_entry["NaV1.5 pIC50"] = nav_pred
        data_entries = [input_data_entry]

        # Compute the scaffold and get the corresponding index
        scaffold = get_scaffold(canonical_input_smiles)

        if scaffold in self.dataset.scaffold_to_idx:
            scaffold_idx = self.dataset.scaffold_to_idx[scaffold]

        else:

            # Determine the most similar scaffold in the training data and the corresponding index
            most_similar_scaffold = get_most_similar_scaffold(
                scaffold, self.transformer_training_data
            )
            scaffold_idx = self.dataset.scaffold_to_idx[most_similar_scaffold]

        # Generate optimized molecules
        generated_smiles = set()
        with tqdm(total=num, desc="Generating optimized molecules") as t:
            while len(generated_smiles) < num:

                # Initialize the sequence with the start token
                sequence = ["[CLS]"]

                # Silence RDKit warnings
                lg = RDLogger.logger()
                lg.setLevel(RDLogger.CRITICAL)

                while (
                    len(sequence) < self.dataset.block_size and "[EOS]" not in sequence
                ):

                    # Create tensors for the input sequence, ADMET properties, and scaffold index
                    token_idx = torch.tensor(
                        [[self.dataset.stoi[s] for s in sequence]], dtype=torch.long
                    ).to(self.device)
                    admet_tensor = torch.tensor(
                        [input_admet_properties], dtype=torch.float
                    ).to(self.device)
                    scaffold_tensor = (
                        torch.tensor([scaffold_idx], dtype=torch.long).to(self.device)
                        if scaffold_idx is not None
                        else None
                    )

                    # Generate the next token
                    logits = self.model(token_idx, admet_tensor, scaffold_tensor)[
                        :, -1, :
                    ]
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    sequence.append(self.dataset.itos[next_token.item()])

                try:

                    # Canonicalize the SMILES string
                    smiles = (
                        "".join(sequence)
                        .replace("[CLS]", "")
                        .replace("[EOS]", "")
                        .replace("<pad>", "")
                    )
                    canonical = Chem.MolToSmiles(
                        Chem.MolFromSmiles(smiles), canonical=True
                    )
                    if canonical and canonical not in generated_smiles:

                        # Filter the generated molecules based on activity against the cardiac ion channels
                        data = prepare_data_for_discriminative_models(
                            canonical,
                            self.transformer_feature_extractor,
                            self.device,
                            add_batch_dim=True,
                        )
                        meets_herg, herg_pred = filter_cardiac_ion_channel_activity(
                            herg_activity,
                            self.herg_classification_model,
                            self.herg_regression_model,
                            data,
                            self.device,
                        )
                        meets_cav, cav_pred = filter_cardiac_ion_channel_activity(
                            cav_activity,
                            self.cav_classification_model,
                            self.cav_regression_model,
                            data,
                            self.device,
                        )
                        meets_nav, nav_pred = filter_cardiac_ion_channel_activity(
                            nav_activity,
                            self.nav_classification_model,
                            self.nav_regression_model,
                            data,
                            self.device,
                        )

                        # Check if the molecule meets all the filters
                        meets_all_filters = all([meets_herg, meets_cav, meets_nav])

                        if meets_all_filters:

                            # Store the SMILES string, corresponding ADMET properties and predictions
                            generated_smiles.add(canonical)
                            admet_values_vector = get_admet_properties(canonical)
                            admet_properties_dict = {
                                label: value
                                for label, value in zip(
                                    admet_labels, admet_values_vector
                                )
                            }
                            entry = {"SMILES": canonical, **admet_properties_dict}
                            if herg_activity is not None:
                                entry["hERG pIC50"] = herg_pred
                            if cav_activity is not None:
                                entry["CaV1.2 pIC50"] = cav_pred
                            if nav_activity is not None:
                                entry["NaV1.5 pIC50"] = nav_pred
                            data_entries.append(entry)

                            t.update(1)

                except:
                    continue

        # Save the results to a CSV file
        df = pd.DataFrame(data_entries)
        df.to_csv(save_path, index=False)

        return df


# Define a class for calculating similarity between the input molecule and the generated molecules
class Similarity_Framework(object):
    """
    A class that provides functionality for calculating molecular descriptors, removing redundant descriptors,
    and calculating similarities between molecules.

    Args:
        generated_smiles (str): The file path to the generated SMILES file.

    Attributes:
        generated_smiles (str): The file path to the generated SMILES file.
        descriptors (list): A list of descriptor names.

    """

    # Define the class constructor
    def __init__(self, generated_smiles):

        # Silence warnings
        warnings.filterwarnings("ignore")

        self.generated_smiles = generated_smiles
        self.descriptors = [desc[0] for desc in Descriptors.descList]

    # Method to calculate molecular descriptors
    def get_descriptor_vector(self, smiles):
        try:

            # Create an RDKit molecule object
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                raise ValueError("Invalid molecule")

            # Return descriptor vector
            return [desc[1](mol) for desc in Descriptors.descList]

        except:
            return [None] * len(self.descriptors)

    # Method to get molecular descriptor vectors for all generated molecules
    def get_descriptors(self, save_path=None):
        df_smiles = pd.read_csv(self.generated_smiles)
        descriptor_data = []
        for smiles in tqdm(
            df_smiles["SMILES"], desc="Calculating molecular descriptors"
        ):

            # Get descriptor vector for each generated molecule
            descriptors = self.get_descriptor_vector(smiles)
            descriptor_data.append([smiles] + descriptors)

        # Save the results to a CSV file
        descriptors_df = pd.DataFrame(
            descriptor_data, columns=["SMILES"] + self.descriptors
        )
        descriptors_df.to_csv(save_path, index=False)

        return descriptors_df

    # Method to identify redundant descriptors
    def get_redundant_descriptors(self, data, mi_threshold):

        # Remove infinite and NaN values
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(axis=1, how="any", inplace=True)

        descriptors = data.columns
        n_descriptors = len(descriptors)

        mi_matrix = np.zeros((n_descriptors, n_descriptors))
        for i in tqdm(range(n_descriptors), desc="Removing redundant descriptors"):
            for j in range(i + 1, n_descriptors):
                col_a = data[descriptors[i]].dropna()
                col_b = data[descriptors[j]].dropna()
                common_index = col_a.index.intersection(col_b.index)

                # Calculate mutual information between descriptor pairs
                mi = mutual_info_score(col_a.loc[common_index], col_b.loc[common_index])
                mi_matrix[i, j] = mi_matrix[j, i] = mi

        mi_matrix = pd.DataFrame(mi_matrix, index=descriptors, columns=descriptors)

        features_to_drop = set()
        for i in range(n_descriptors):
            for j in range(i + 1, n_descriptors):
                if mi_matrix.iloc[i, j] > mi_threshold:

                    # Drop the descriptor with the lower mean mutual information with all other descriptors
                    if mi_matrix.iloc[i, :].mean() > mi_matrix.iloc[j, :].mean():
                        features_to_drop.add(descriptors[i])

                    else:
                        features_to_drop.add(descriptors[j])

        return features_to_drop

    # Method to remove redundant descriptors
    def remove_redundant_descriptors(
        self, descriptors_path, mi_threshold=0.9, save_path=None
    ):
        descriptors_df = pd.read_csv(descriptors_path)

        # Determine the descriptors to drop
        features_to_drop = self.get_redundant_descriptors(
            descriptors_df.drop(columns=["SMILES"]), mi_threshold
        )

        # Remove redundant descriptors
        filtered_df = descriptors_df.drop(columns=features_to_drop)

        # Save the results to a CSV file
        filtered_df.to_csv(save_path, index=False)

        return filtered_df

    # Method to calculate similarities between the input molecule and the generated molecules
    def calculate_similarities(self, descriptors_file, generation_file, save_path=None):

        # Load the descriptor data
        descriptors_df = pd.read_csv(descriptors_file)
        descriptor_values = descriptors_df.iloc[:, 1:].values

        # Normalize the descriptor values
        scaler = StandardScaler()
        normalized_descriptors = scaler.fit_transform(descriptor_values.T).T
        normalized_descriptors_df = pd.DataFrame(
            normalized_descriptors, columns=descriptors_df.columns[1:]
        )
        normalized_descriptors_df.insert(
            0, descriptors_df.columns[0], descriptors_df[descriptors_df.columns[0]]
        )

        # Load the ADMET property data
        admet_df = pd.read_csv(generation_file)

        admet_properties = [
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
        ]
        pic50_columns = [
            col for col in admet_df.columns if col not in admet_properties + ["SMILES"]
        ]
        final_columns_order = (
            ["SMILES"]
            + ["Similarity"]
            + pic50_columns
            + admet_properties
            + list(normalized_descriptors_df.columns[1:])
        )

        final_df = pd.concat(
            [admet_df, normalized_descriptors_df.drop(columns=["SMILES"])],
            axis=1,
            join="inner",
        )

        # Define the input molecule descriptor vector
        input_smiles = descriptors_df["SMILES"].iloc[0]
        input_descriptor_row = normalized_descriptors_df[
            normalized_descriptors_df["SMILES"] == input_smiles
        ]
        input_descriptor_vector = input_descriptor_row.drop(
            columns=["SMILES"]
        ).values.reshape(1, -1)

        # Calculate the similarity between the input molecule descriptor vector and the generated molecule descriptor vectors
        cosine_similarities = cosine_similarity(
            normalized_descriptors_df.iloc[:, 1:].values, input_descriptor_vector
        ).flatten()
        similarity_series = pd.Series(
            cosine_similarities, index=normalized_descriptors_df["SMILES"]
        )

        final_df["Similarity"] = similarity_series.values
        final_df = final_df[final_columns_order]
        final_df_sorted = final_df.sort_values(by="Similarity", ascending=False)
        final_df_sorted.reset_index(drop=True, inplace=True)

        # Save the results to a CSV file
        final_df_sorted.to_csv(save_path, index=False)
        print(
            f"\nSimilarity calculations completed. Results saved to {save_path}. See DataFrame below: ⬇️"
        )

        return final_df_sorted


# Define a function to run the complete CardioGenAI framework
def optimize_cardiotoxic_drug(
    input_smiles="O=c1[nH]c2ccccc2n1C1CCN(CCCC(c2ccc(F)cc2)c2ccc(F)cc2)CC1",
    n_generations=100,
    herg_activity=(0, 6.0),
    nav_activity=None,
    cav_activity=None,
    device="gpu",
    autoregressive_transformer_params="model_parameters/transformer_model_parameters/Autoregressive_Transformer_parameters.pt",
    bidirectional_transformer_params="model_parameters/transformer_model_parameters/Bidirectional_Transformer_parameters.pt",
    transformer_training_data="data/prepared_transformer_datasets/prepared_transformer_data.csv",
    herg_regression_params="model_parameters/discriminative_model_parameters/hERG_Regression_parameters.pt",
    nav_regression_params="model_parameters/discriminative_model_parameters/Nav_Regression_parameters.pt",
    cav_regression_params="model_parameters/discriminative_model_parameters/Cav_Regression_parameters.pt",
    herg_classification_params="model_parameters/discriminative_model_parameters/hERG_Classification_parameters.pt",
    nav_classification_params="model_parameters/discriminative_model_parameters/Nav_Classification_parameters.pt",
    cav_classification_params="model_parameters/discriminative_model_parameters/Cav_Classification_parameters.pt",
):
    """
    Optimize a cardiotoxic drug using a generative model and calculate its similarities with existing drugs.

    Args:
        input_smiles (str): The input SMILES representation of the drug molecule. Default is a specific SMILES string.
        n_generations (int): The number of generations to perform for optimizing the drug. Default is 100.
        herg_activity (tuple): The range of hERG activity values to optimize for. Default is (0, 6.0).
        nav_activity (tuple): The range of Nav activity values to optimize for. Default is None.
        cav_activity (tuple): The range of Cav activity values to optimize for. Default is None.
        device (str): The device to use for the optimization. Must be either 'cpu' or 'gpu'. Default is 'gpu'.
        autoregressive_transformer_params (str): The path to the autoregressive transformer model parameters file.
        bidirectional_transformer_params (str): The path to the bidirectional transformer model parameters file.
        transformer_training_data (str): The path to the transformer training data file.
        herg_regression_params (str): The path to the hERG regression model parameters file.
        nav_regression_params (str): The path to the Nav regression model parameters file.
        cav_regression_params (str): The path to the Cav regression model parameters file.
        herg_classification_params (str): The path to the hERG classification model parameters file.
        nav_classification_params (str): The path to the Nav classification model parameters file.
        cav_classification_params (str): The path to the Cav classification model parameters file.

    Returns:
        dict: A dictionary containing the results of the optimization and similarity calculations.
    """

    assert input_smiles is not None, "Please provide input SMILES"
    assert (
        autoregressive_transformer_params is not None
    ), "Please provide path to autoregressive transformer model"
    assert (
        bidirectional_transformer_params is not None
    ), "Please provide path to bidirectional transformer model"
    assert (
        transformer_training_data is not None
    ), "Please provide path to transformer training data"
    assert any(
        [
            herg_regression_params,
            nav_regression_params,
            cav_regression_params,
            herg_classification_params,
            nav_classification_params,
            cav_classification_params,
        ]
    ), "Please provide at least one discriminative model"
    assert any(
        [herg_activity, nav_activity, cav_activity]
    ), "Please provide at least one cardiac ion channel activity for which to optimize"
    assert n_generations is not None, "Please provide number of generations"
    assert device in [
        "cpu",
        "gpu",
    ], "Invalid device value. Must be either 'cpu' or 'gpu'"

    # Define the generation object
    generator = Generation_Framework(
        autoregressive_transformer_params=autoregressive_transformer_params,
        bidirectional_transformer_params=bidirectional_transformer_params,
        transformer_training_data=transformer_training_data,
        herg_regression_params=herg_regression_params,
        nav_regression_params=nav_regression_params,
        cav_regression_params=cav_regression_params,
        herg_classification_params=herg_classification_params,
        nav_classification_params=nav_classification_params,
        cav_classification_params=cav_classification_params,
        device=device,
    )

    # Generate optimized molecules
    generator.generate_smiles(
        input_smiles=input_smiles,
        num=n_generations,
        herg_activity=herg_activity,
        cav_activity=cav_activity,
        nav_activity=nav_activity,
        save_path="results/optimization_results/optimized_drugs.csv",
    )

    # Define the similarity object
    similator = Similarity_Framework(
        generated_smiles="results/optimization_results/optimized_drugs.csv"
    )

    # Calculate molecular descriptor vectors
    similator.get_descriptors(
        save_path="results/optimization_results/optimized_drugs_descriptors.csv"
    )

    # Remove redundant descriptors
    similator.remove_redundant_descriptors(
        descriptors_path="results/optimization_results/optimized_drugs_descriptors.csv",
        save_path="results/optimization_results/optimized_drugs_descriptors_nonredundant.csv",
    )

    # Calculate similarities between the input molecule descriptor vector and the generated molecule descriptor vectors
    results = similator.calculate_similarities(
        descriptors_file="results/optimization_results/optimized_drugs_descriptors_nonredundant.csv",
        generation_file="results/optimization_results/optimized_drugs.csv",
        save_path="results/optimization_results/optimized_drugs_similarities.csv",
    )

    return results
