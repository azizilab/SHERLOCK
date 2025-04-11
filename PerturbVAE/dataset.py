import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class PerturbDataset(Dataset):
    def __init__(self, anndata):
        """
        Initialize the dataset with an AnnData object with top_sg for pertubation and treatment for condition.

        Args:
            anndata: AnnData object containing the data matrix (X) and metadata (obs).
        """
        self.X = anndata.X.toarray() if hasattr(anndata.X, "toarray") else anndata.X
        self.X = self.X.astype(np.float32)

        # Extract perturbation (P) and condition (C) from obs
        self.P = anndata.obs['top_sg'].values
        self.C = anndata.obs['treatment'].values

        # Create dictionaries mapping unique perturbations and conditions to indices
        self.perturbation_dict = {p: i for i, p in enumerate(np.unique(self.P))}
        self.condition_dict = {c: i for i, c in enumerate(np.unique(self.C))}

        # Check for 'NA' or NaN in perturbation_dict and condition_dict
        if 'NA' in self.perturbation_dict or 'NA' in self.condition_dict:
            raise ValueError("The dataset contains 'NA' in perturbation or condition attributes.")
        if any(pd.isna(key) for key in self.perturbation_dict.keys()) or any(pd.isna(key) for key in self.condition_dict.keys()):
            raise ValueError("The dataset contains NaN values in perturbation or condition attributes.")

        # Convert P and C to indices
        self.P_indices = np.array([self.perturbation_dict[p] for p in self.P])
        self.C_indices = np.array([self.condition_dict[c] for c in self.C])

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple (X, P, C) where:
                - X: The log1p-transformed data for the sample.
                - P: The index of the perturbation for the sample.
                - C: The index of the condition for the sample.
        """
        X = self.X[idx]
        P = self.P_indices[idx]
        C = self.C_indices[idx]
        return X, P, C