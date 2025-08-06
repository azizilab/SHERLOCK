import torch
from torch.utils.data import Dataset, Sampler, DataLoader
import numpy as np
import pandas as pd
import random
import itertools
from collections import defaultdict

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
    




class MultiClassBatchSampler(Sampler[list[int]]):
    """
    Yields batches that comprise *all* examples from `n_classes_per_batch`
    randomly-chosen class labels.

    Parameters
    ----------
    labels               : 1-D array-like of ints, length == dataset
    n_classes_per_batch  : how many distinct classes to pack into one batch
    shuffle_within_class : shuffle order of indices inside each class
    drop_last            : drop the final (smaller) batch of classes if the
                           total #classes isn't divisible by n_classes_per_batch
    """
    def __init__(self, labels,
                 n_classes_per_batch: int,
                 shuffle_within_class: bool = False,
                 drop_last: bool = False):
        self.labels   = np.asarray(labels)
        self.k        = int(n_classes_per_batch)
        self.shuffle_within = shuffle_within_class
        self.drop_last = drop_last

        # build {class_id: [idx0, idx1, …]}
        self.class_to_idxs = defaultdict(list)
        for idx, lab in enumerate(self.labels):
            self.class_to_idxs[int(lab)].append(idx)

        # keep only non-empty classes
        self.classes = list(self.class_to_idxs.keys())

    def __len__(self):
        n_groups = len(self.classes) // self.k
        if not self.drop_last and len(self.classes) % self.k:
            n_groups += 1
        return n_groups

    def __iter__(self):
        # shuffle class order every epoch
        random.shuffle(self.classes)

        # slice classes into groups of k
        for i in range(0, len(self.classes), self.k):
            group = self.classes[i : i + self.k]
            if len(group) < self.k and self.drop_last:
                break

            # gather indices from each class
            batch_idxs = []
            for c in group:
                idxs = self.class_to_idxs[c]
                if self.shuffle_within:
                    random.shuffle(idxs)
                batch_idxs.extend(idxs)

            yield batch_idxs