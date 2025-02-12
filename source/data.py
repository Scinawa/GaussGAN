import pickle
from typing import NamedTuple

import numpy as np
import torch
from lightning import LightningDataModule
from rdkit import Chem, RDLogger
from torch.utils.data import DataLoader, Dataset, random_split
from torch_geometric.data import Data
from torchdrug.data import Molecule
from tqdm import tqdm

from .metrics import ALL_METRICS
from .nn import QuantumNoise

# from lightning.pytorch.trainer.connectors.data_connector import DataConnector

# import warnings
# # Add this before any other imports
# warnings.filterwarnings(
#     "ignore",
#     message="*num_workers*",
#     category=UserWarning
# )

# class GaussianDataset(Dataset):

#     def __init__(self, n_samples, mean1, cov1, mean2, cov2):


#         # Generate 500 points from each distribution
#         points1 = np.random.multivariate_normal(mean1, cov1, n_samples)
#         points2 = np.random.multivariate_normal(mean2, cov2, n_samples)

#         # Combine the points into a single dataset
#         self.dataset = np.vstack((points1, points2))


#     def __len__(self):
#         """Returns the number of valid molecules in the dataset."""
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         """Retrieve a sample from the dataset by index."""
#         return self.dataset[idx]


#     def save(self, filename):
#         """Saves the dataset to a pickle file."""
#         with open(filename, "wb") as f:
#             pickle.dump(self, f)
#         print(f"Dataset saved to {filename}.")

#     @classmethod
#     def load(cls, filename):
#         """Loads the dataset from a pickle file."""
#         with open(filename, "rb") as f:
#             dataset = pickle.load(f)
#         print(f"Dataset loaded from {filename}.")
#         return dataset


class GaussianDataModule(LightningDataModule):
    def __init__(
        self,
        dataset,
        *,
        batch_size=32,
        train_test_val_split=(0.8, 0.1, 0.1),
    ):
        super().__init__()
        # Save the batch size as a hyperparameter
        self.save_hyperparameters("batch_size")
        # Save the dataset
        self.dataset = dataset
        self.train_test_val_split = train_test_val_split
        # Initialize the train, validation, and test datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # This method should only run on 1 GPU/TPU in distributed settings,
        # thus we do not need to set anything related to the dataset itself here,
        # since it's done in setup() which is called on every GPU/TPU.
        pass

    def setup(self, stage=None):
        # Calculate split sizes based on the provided tuple ratios
        train_size = int(self.train_test_val_split[0] * len(self.dataset))
        val_size = int(self.train_test_val_split[1] * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        # Perform the split
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        # Returns the training dataloader.
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=0 #collate_fn=collate_fn,
        )

    def val_dataloader(self):
        # Returns the validation dataloader.
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            #collate_fn=collate_fn,
        )

    def test_dataloader(self):
        # Returns the testing dataloader.
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            #collate_fn=collate_fn,
        )

class QNoiseData(LightningDataModule):
    def __init__(self, 
                 n_samples, 
                 batch_size=32, 
                 val_test_split=(0.2,0.2), 
                 num_qubits=1, 
                 num_layers=1):
        super().__init__()
        self.save_hyperparameters()
        self.n_samples = n_samples
        self.generator = QuantumNoise(num_qubits, num_layers)
        self.batch_size = batch_size
        self.val_test_split = val_test_split
        self.x, self.y = self.generate_data()
        
    def generate_data(self):
        x = []
        y = []
        for _ in range(self.n_samples):
            z1 = np.random.uniform(-1, 1)
            z2 = np.random.uniform(-1, 1)
            output = self.generator.gen_circuit_with_input(z1, z2)
            x.append(torch.tensor([z1, z2], dtype=torch.float32))
            y.append(output)
        y = torch.tensor(y, dtype=torch.float32).detach()
        return x, y

    def setup(self, stage=None):
        train_size = int((1 - sum(self.val_test_split)) * self.n_samples)
        val_size = int(self.val_test_split[0] * self.n_samples)
        test_size = self.n_samples - train_size - val_size
        
        self.train_data = self.x[:train_size], self.y[:train_size]
        self.val_data = self.x[train_size:train_size + val_size], self.y[train_size:train_size + val_size]
        self.test_data = self.x[-test_size:], self.y[-test_size:]
    
    def train_dataloader(self):
        x, y = self.train_data
        return DataLoader(list(zip(x, y)), batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        x, y = self.val_data
        return DataLoader(list(zip(x, y)), batch_size=self.batch_size)
    
    def test_dataloader(self):
        x, y = self.test_data
        return DataLoader(list(zip(x, y)), batch_size=self.batch_size)