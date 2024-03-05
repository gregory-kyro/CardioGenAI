# Import necessary libraries
import math
import pandas as pd
import random
import re
import torch
from torch.cuda.amp import GradScaler
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# In order to train the transformers with the SophiaG optimizer, please clone the SophiaG 
# repository from here: <https://github.com/Liuhong99/Sophia.git> and then import the 
# SophiaG optimizer: from Sophia.sophia import SophiaG


# Define a class for the transformer dataset
class Transformer_Dataset(Dataset):
    """
    Dataset class for the Transformer model.

    Args:
        mode (str): The mode of the dataset. Must be either 'Autoregressive' or 'Bidirectional'.
        data_path (str): The path to the data file.
        block_size (int): The block size for padding the sequences.

    Attributes:
        smiles (pd.Series): The SMILES data from the dataset.
        mode (str): The mode of the dataset.
        smiles_regex (re.Pattern): The regular expression pattern for tokenizing SMILES strings.
        admet_props (pd.DataFrame): The ADMET properties from the dataset.
        scaffolds (pd.Series): The scaffold information from the dataset.
        scaffold_to_idx (dict): A mapping of scaffold strings to their corresponding indices.
        num_scaffolds (int): The number of unique scaffolds in the dataset.
        vocab (list): The vocabulary of special tokens and characters.
        stoi (dict): A mapping of characters to their corresponding indices in the vocabulary.
        itos (dict): A mapping of indices to their corresponding characters in the vocabulary.
        block_size (int): The block size for padding the sequences.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the item at the given index.

    """

    # Define the class constructor
    def __init__(self, mode="Autoregressive", data_path=None, block_size=133):

        assert mode in [
            "Autoregressive",
            "Bidirectional",
        ], "Mode must be either Autoregressive or Bidirectional"
        self.mode = mode

        # Retrieve the SMILES strings from the data file
        data = pd.read_csv(data_path)
        self.smiles = data["SMILES"]

        if self.mode == "Autoregressive":

            # Tokenize the SMILES strings
            self.smiles_regex = re.compile(
                "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]|<pad>|[CLS]|[EOS])"
            )
            special_tokens = {"<pad>", "[CLS]", "[EOS]"}

            # Retrieve the ADMET properties from the data file
            self.admet_props = data[
                [
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
            ]

            # Retrieve the scaffold information from the data file and create a mapping to indices
            self.scaffolds = data["scaffold"].fillna("scaffold_placeholder")
            unique_scaffolds = sorted(set(self.scaffolds))
            self.scaffold_to_idx = {s: i for i, s in enumerate(unique_scaffolds)}

            self.num_scaffolds = len(unique_scaffolds)

        else:

            # Tokenize the SMILES strings
            self.smiles_regex = re.compile(
                "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]|<MASK>|<pad>|[CLS]|[EOS])"
            )
            special_tokens = {"<MASK>", "<pad>", "[CLS]", "[EOS]"}

        # Build the vocabulary
        characters = {
            ch
            for smile in self.smiles
            for ch in self.smiles_regex.findall(smile.strip())
        }
        self.vocab = sorted(list(special_tokens | characters))

        # Create mappings for the vocabulary
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

        self.block_size = 2 + block_size

    # Method to return the length of the dataset
    def __len__(self):
        return len(self.smiles)

    # Method to return the item at the given index
    def __getitem__(self, idx):

        # Tokenize the SMILES string and pad it to the block size
        smiles = "[CLS]" + self.smiles[idx].strip() + "[EOS]"
        smiles_tokens = self.smiles_regex.findall(smiles)
        smiles += "<pad>" * (self.block_size - len(smiles_tokens))

        if self.mode == "Autoregressive":
            # Tokenize the SMILES string and retrieve the ADMET properties and scaffold index
            input_sequence = [
                self.stoi[s] for s in self.smiles_regex.findall(smiles)[:-1]
            ]
            target_sequence = [
                self.stoi[s] for s in self.smiles_regex.findall(smiles)[1:]
            ]
            admet_properties = self.admet_props.iloc[idx].values
            scaffold_idx = self.scaffold_to_idx[self.scaffolds[idx]]

            return (
                torch.tensor(input_sequence, dtype=torch.long),
                torch.tensor(target_sequence, dtype=torch.long),
                torch.tensor(admet_properties, dtype=torch.float),
                torch.tensor(scaffold_idx, dtype=torch.long),
            )

        else:

            # Retrieve the token indices
            true_token_idx = [self.stoi[s] for s in self.smiles_regex.findall(smiles)]

            # Mask the tokens
            mask_idx = []
            for s in range(len(smiles_tokens)):
                if random.random() < 0.15:
                    mask_idx.append(False)
                    num = random.random()
                    if num >= 0.2:
                        smiles_tokens[s] = "<MASK>"
                    elif num >= 0.1:
                        smiles_tokens[s] = self.vocab[
                            int(random.random() * len(self.vocab))
                        ]
                else:
                    mask_idx.append(True)

            # Identify the masked tokens
            mask_idx += [True] * (self.block_size - len(mask_idx))
            masked_smiles = "".join(smiles_tokens)

            # Pad to the block size
            masked_smiles += "<pad>" * (
                self.block_size - len(self.smiles_regex.findall(masked_smiles))
            )
            masked_token_idx = [
                self.stoi[s] for s in self.smiles_regex.findall(masked_smiles)
            ]

            return (
                torch.tensor(masked_token_idx, dtype=torch.long),
                torch.tensor(true_token_idx, dtype=torch.long),
                torch.tensor(mask_idx),
            )


# Define the transformer model
class Transformer_Model(nn.Module):
    """
    Transformer model for sequence generation.

    Args:
        mode (str): Mode of the model, either 'Autoregressive' or 'Bidirectional'.
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimension of the token embeddings.
        block_size (int): Size of the input sequence.
        n_layers (int): Number of transformer blocks.
        admet_dim (int): Dimension of the admet properties.
        num_scaffolds (int): Number of scaffolds.
        extract_features (bool): Whether to extract features instead of generating output.

    Attributes:
        mode (str): Mode of the model.
        token_embed (nn.Embedding): Token embedding layer.
        position_embed (nn.Parameter): Positional embedding layer.
        dropout (nn.Dropout): Dropout layer.
        blocks (nn.ModuleList): List of transformer blocks.
        layer_norm (nn.LayerNorm): Layer normalization layer.
        output (nn.Linear): Output layer.
        extract_features (bool): Whether to extract features instead of generating output.
        admet_embed (nn.Linear): Admet embedding layer (only for 'Autoregressive' mode).
        scaffold_embed (nn.Embedding): Scaffold embedding layer (only for 'Autoregressive' mode).
        type_embed (nn.Embedding): Type embedding layer (only for 'Bidirectional' mode).

    Methods:
        forward(idx, admet_props=None, scaffold_idx=None, return_attention_weights=False):
            Forward pass of the model.

    """

    # Define the class constructor
    def __init__(
        self,
        mode="Autoregressive",
        vocab_size=None,
        embed_dim=256,
        block_size=133,
        n_layers=8,
        admet_dim=10,
        num_scaffolds=1143740,
        extract_features=False,
    ):

        super().__init__()

        assert mode in [
            "Autoregressive",
            "Bidirectional",
        ], "Mode must be either Autoregressive or Bidirectional"
        self.mode = mode

        self.extract_features = extract_features

        # Define token and position embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = nn.Parameter(torch.zeros(1, block_size, embed_dim))

        self.dropout = nn.Dropout(0.1)
        self.blocks = nn.ModuleList(
            [
                Transformer_Block(embed_dim=embed_dim, mode=self.mode)
                for _ in range(n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size, bias=True)

        if self.mode == "Autoregressive":

            # Define ADMET and scaffold embeddings
            self.admet_embed = nn.Linear(admet_dim, embed_dim)
            self.scaffold_embed = nn.Embedding(num_scaffolds, embed_dim)

        else:

            # Define type embedding
            self.type_embed = nn.Embedding(2, embed_dim)

    # Method to perform the forward pass
    def forward(
        self, idx, admet_props=None, scaffold_idx=None, return_attention_weights=False
    ):

        # Get batch and time dimensions
        B, T = idx.size()

        # Define token and position embeddings
        token_embeddings = self.token_embed(idx)
        position_embeddings = self.position_embed[:, :T, :]

        if self.mode == "Autoregressive":

            # Define ADMET and scaffold embeddings
            admet_embeddings = (
                self.admet_embed(admet_props).unsqueeze(1).expand(-1, T, -1)
            )
            scaffold_embeddings = (
                self.scaffold_embed(scaffold_idx).unsqueeze(1).expand(-1, T, -1)
            )

            # Add token, position, ADMET, and scaffold embeddings
            x = self.dropout(
                token_embeddings
                + position_embeddings
                + admet_embeddings
                + scaffold_embeddings
            )

        else:

            # Define type embeddings
            type_embeddings = self.type_embed(
                torch.ones((B, T), dtype=torch.long, device=idx.device)
            )

            # Add token, position, and type embeddings
            x = self.dropout(token_embeddings + position_embeddings + type_embeddings)

        # Perform the forward pass through the transformer blocks
        attention_weights_list = []
        for block in self.blocks:

            if return_attention_weights:
                x, attention_weights = block(x, return_attention_weights=True)
                attention_weights_list.append(attention_weights)

            else:
                x = block(x)

        # Apply layer normalization
        x = self.layer_norm(x)

        if self.extract_features:
            return x.detach()

        # Generate the output logits
        logits = self.output(x)

        if return_attention_weights:
            return logits, attention_weights_list

        else:
            return logits


# Define the transformer block
class Transformer_Block(nn.Module):
    """
    Transformer Block class.

    Args:
        embed_dim (int): The dimensionality of the input embeddings. Default is 256.
        mode (str): The mode of the transformer block. Default is 'Autoregressive'.

    Attributes:
        mode (str): The mode of the transformer block.
        layer_norm1 (nn.LayerNorm): Layer normalization module.
        layer_norm2 (nn.LayerNorm): Layer normalization module.
        attention (Self_Attention): Self-attention module.
        mlp (nn.Sequential): Multi-layer perceptron module.

    Methods:
        forward(x, return_attention_weights=False): Performs forward pass of the transformer block.

    """

    # Define the class constructor
    def __init__(self, embed_dim=256, mode="Autoregressive"):
        super().__init__()
        self.mode = mode

        # Define layer normalization modules
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        # Define self-attention module
        self.attention = Self_Attention(embed_dim=embed_dim, mode=self.mode)

        # Define multi-layer perceptron module
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(0.1),
        )

    # Method to perform the forward pass
    def forward(self, x, return_attention_weights=False):

        if return_attention_weights:
            y, attention_weights = self.attention(
                self.layer_norm1(x), return_attention_weights=True
            )
            x = x + y
            x = x + self.mlp(self.layer_norm2(x))
            return x, attention_weights

        else:
            y = self.attention(self.layer_norm1(x))
            x = x + y
            x = x + self.mlp(self.layer_norm2(x))
            return x


# Define the self-attention module
class Self_Attention(nn.Module):
    """
    Self-Attention module that performs attention mechanism on the input sequence.

    Args:
        embed_dim (int): The dimensionality of the input embeddings.
        n_heads (int): The number of attention heads.
        mode (str): The mode of operation. Can be 'Autoregressive' or 'Parallel'.

    Attributes:
        mode (str): The mode of operation.
        query (nn.Linear): Linear layer for computing the query.
        key (nn.Linear): Linear layer for computing the key.
        value (nn.Linear): Linear layer for computing the value.
        dropout (nn.Dropout): Dropout layer for regularization.
        projection (nn.Linear): Linear layer for projecting the output.
        n_heads (int): The number of attention heads.

    Methods:
        forward(x, return_attention_weights=False): Performs forward pass of the self-attention module.

    """

    # Define the class constructor
    def __init__(self, embed_dim=256, n_heads=8, mode="Autoregressive"):
        super().__init__()
        self.mode = mode

        # Define linear layers for query, key, and value vectors
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(0.1)
        self.projection = nn.Linear(embed_dim, embed_dim)

        self.n_heads = n_heads

    # Method to perform the forward pass
    def forward(self, x, return_attention_weights=False):
        # Get batch, time, and channel dimensions
        B, T, C = x.size()

        # Compute query, key, and value vectors
        q = self.query(x).view(B, T, self.n_heads, C // self.n_heads)
        k = self.key(x).view(B, T, self.n_heads, C // self.n_heads)
        v = self.value(x).view(B, T, self.n_heads, C // self.n_heads)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        scale_factor = 1 / math.sqrt(C // self.n_heads)
        attn_bias = torch.zeros(T, T, dtype=q.dtype, device=q.device)

        if self.mode == "Autoregressive":
            # Create a mask for the upper triangular part of the attention matrix
            temp_mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril(
                diagonal=0
            )
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

        # Compute the attention weights
        attn_weight = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = self.dropout(attn_weight)

        # Multiply the attention weights by the value vectors
        y = torch.matmul(attn_weight, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Map the output to the embedding dimension
        y = self.projection(y)

        y = self.dropout(y)

        if return_attention_weights:
            return y, attn_weight

        else:
            return y


# Define a function to train the transformer model
def train_transformer(
    mode="Autoregressive",
    training_data=None,
    checkpoint_path=None,
    learning_rate=3e-4,
    weight_decay=0.1,
    batch_size=512,
    epochs=100,
    validation_split=0.05,
    device="gpu",
    verbose=False,
):
    """
    Trains a Transformer model for a given mode (Autoregressive or Bidirectional) using the provided training data.

    Args:
        mode (str, optional): The mode of the Transformer model. Must be either 'Autoregressive' or 'Bidirectional'. Defaults to 'Autoregressive'.
        training_data (str, optional): The path to the training data. Defaults to None.
        checkpoint_path (str, optional): The path to save the model checkpoints. Defaults to None.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 3e-4.
        weight_decay (float, optional): The weight decay for the optimizer. Defaults to 0.1.
        batch_size (int, optional): The batch size for training. Defaults to 512.
        epochs (int, optional): The number of training epochs. Defaults to 100.
        validation_split (float, optional): The fraction of data to use for validation. Defaults to 0.05.
        device (str, optional): The device to use for training. Must be either 'cpu' or 'gpu'. Defaults to 'gpu'.
        verbose (bool, optional): Whether to print training progress. Defaults to False.

    Raises:
        AssertionError: If the device is not 'cpu' or 'gpu'.
        AssertionError: If the mode is not 'Autoregressive' or 'Bidirectional'.

    Returns:
        None
    """
    assert mode in [
        "Autoregressive",
        "Bidirectional",
    ], "Mode must be either Autoregressive or Bidirectional"

    assert device in ["cpu", "gpu"], "Device must be either 'cpu' or 'gpu'."
    device = torch.device(
        "cuda:0" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
    )

    # Load the training data
    dataset = Transformer_Dataset(mode=mode, data_path=training_data)

    # Split the data into training and validation sets
    total_size = len(dataset)
    val_size = int(validation_split * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, shuffle=True, pin_memory=True, batch_size=batch_size
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, pin_memory=True, batch_size=batch_size
    )

    (
        print(
            "Successfully loaded {} training samples and {} validation samples".format(
                train_size, val_size
            )
        )
        if verbose
        else None
    )

    # Define the model
    model = Transformer_Model(
        mode=mode,
        vocab_size=len(dataset.vocab),
        block_size=dataset.block_size,
        num_scaffolds=dataset.num_scaffolds,
    ).to(device)

    print("Successfully built {} model".format(mode)) if verbose else None

    optimizer = SophiaG(
        model.parameters(),
        lr=learning_rate,
        betas=(0.965, 0.99),
        rho=0.04,
        weight_decay=weight_decay,
    )
    scaler = GradScaler()
    loss_function = (
        CrossEntropyLoss(ignore_index=-1)
        if mode == "Autoregressive"
        else CrossEntropyLoss(ignore_index=dataset.stoi["<pad>"])
    )

    epoch_train_losses = []
    epoch_val_losses = []

    # Train the model
    for epoch in range(epochs):

        model.train()
        total_loss = 0

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{epochs}",
        )
        for batch_id, batch in pbar:

            # Zero the gradients
            optimizer.zero_grad()

            if mode == "Autoregressive":
                # Unpack the batch
                x, y, admet_props, scaffold_idx = [item.to(device) for item in batch]
            else:
                # Unpack the batch
                x, y, mask_idx = [item.to(device) for item in batch]

            with torch.cuda.amp.autocast():
                if mode == "Autoregressive":

                    # Predict the logits
                    logits = model(x, admet_props, scaffold_idx)

                    # Compute the loss
                    loss = loss_function(logits.view(-1, logits.size(-1)), y.view(-1))

                else:

                    # Predict the logits
                    logits = model(x)

                    # Compute the loss
                    y[mask_idx] = -1
                    loss = loss_function(logits.view(-1, logits.size(-1)), y.view(-1))

            # Backpropagate the gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_description(
                f"Epoch: {epoch+1}/{epochs}; Training Loss: {total_loss/(batch_id+1):.4f}"
            )

        avg_train_loss = total_loss / len(train_loader)
        epoch_train_losses.append(avg_train_loss)

        # Validate the model
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if mode == "Autoregressive":

                    # Unpack the batch
                    x, y, admet_props, scaffold_idx = [
                        item.to(device) for item in batch
                    ]

                    # Predict the logits
                    logits = model(x, admet_props, scaffold_idx)

                    # Compute the loss
                    loss = loss_function(logits.view(-1, logits.size(-1)), y.view(-1))

                else:

                    # Unpack the batch
                    x, y, mask_idx = [item.to(device) for item in batch]

                    # Predict the logits
                    logits = model(x)

                    # Compute the loss
                    y[mask_idx] = -1
                    loss = loss_function(logits.view(-1, logits.size(-1)), y.view(-1))

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        epoch_val_losses.append(avg_val_loss)

        # Save the model checkpoint
        checkpoint_dict = {
            "model_state_dict": model.state_dict(),
            "training_losses": epoch_train_losses,
            "validation_losses": epoch_val_losses,
            "epoch": epoch,
        }

        torch.save(checkpoint_dict, checkpoint_path)


# Define a class for the transformer feature extractor
class Transformer_Feature_Extractor(torch.nn.Module):
    """
    A class representing a Transformer feature extractor.

    Args:
        model_parameters (str): The path to the model parameters file.
        training_data (str): The path to the training data.
        device (str, optional): The device to use for computation. Must be either 'cpu' or 'gpu'. Defaults to 'gpu' if available.

    Attributes:
        device (torch.device): The device used for computation.
        dataset (Transformer_Dataset): The dataset used for training.
        model (Transformer_Model): The Transformer model used for feature extraction.

    Methods:
        extract_features: Extracts features from input smiles.

    """

    # Define the class constructor
    def __init__(self, model_parameters, training_data, device="gpu"):
        super().__init__()
        assert device in ["cpu", "gpu"], "Device must be either 'cpu' or 'gpu'."
        self.device = torch.device(
            "cuda:0" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
        )

        # Load the dataset and model
        self.dataset = Transformer_Dataset(
            mode="Bidirectional", data_path=training_data
        )
        self.model = Transformer_Model(
            mode="Bidirectional",
            vocab_size=len(self.dataset.vocab),
            block_size=self.dataset.block_size,
            extract_features=True,
        ).to(self.device)

        # Load the model parameters
        checkpoint = torch.load(model_parameters, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)

        self.model.eval()

    # Method to extract features from input smiles
    def extract_features(self, smiles):

        # Tokenize the SMILES string and pad it to the block size
        smiles = "[CLS]" + smiles.strip() + "[EOS]"
        smiles_tokens = self.dataset.smiles_regex.findall(smiles)
        smiles += "<pad>" * (self.dataset.block_size - len(smiles_tokens))
        token_idx = (
            torch.tensor(
                [
                    self.dataset.stoi[s]
                    for s in self.dataset.smiles_regex.findall(smiles)
                ],
                dtype=torch.long,
            )
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            # Extract the features
            features = self.model(token_idx).squeeze(0)

        return features
