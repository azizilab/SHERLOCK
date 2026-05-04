import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from .._configs import get_config

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

class _PerturbCollateFn:
    """Picklable collate callable for PerturbMatchingDataset (required for num_workers > 0)."""

    def __init__(self, n_ntc, X_ntc):
        self.n_ntc = n_ntc
        self.X_ntc = X_ntc

    def __call__(self, batch):
        n_ntc = self.n_ntc
        X_ntc = self.X_ntc
        rng   = np.random.default_rng()

        x_p_list, p_list, c_list = zip(*batch)
        x_p = torch.from_numpy(np.stack(x_p_list).astype(np.float32))
        c   = torch.tensor(list(c_list), dtype=torch.long)

        if isinstance(p_list[0], np.ndarray):
            p = torch.from_numpy(np.stack(p_list))
        else:
            p = torch.tensor(list(p_list), dtype=torch.long)

        unique_conds = np.unique(list(c_list))
        ntc_chunks, c_ntc_chunks = [], []
        for cond_i in unique_conds:
            pool = X_ntc[int(cond_i)]
            n    = min(n_ntc, len(pool))
            idx  = rng.choice(len(pool), size=n, replace=False)
            ntc_chunks.append(pool[idx].astype(np.float32))
            c_ntc_chunks.extend([int(cond_i)] * n)

        x_ntc = torch.from_numpy(np.concatenate(ntc_chunks, axis=0))
        c_ntc = torch.tensor(c_ntc_chunks, dtype=torch.long)

        return x_p, x_ntc, p, c, c_ntc


class PerturbMatchingDataset(Dataset):
    """
    Dataset of perturbed cells paired with a batch of NTC cells.

    • X_p       – all non-NTC cells   (n_p × G)
    • X_ntc     – NTC cells per condition, kept for eval/_counterfactual_effect_size
    • X_ntc_flat – all NTC cells concatenated, sampled by get_collate_fn()

    Use get_collate_fn() to build the DataLoader collate function, which adds
    n_ntc randomly-sampled (without replacement) NTC cells to each batch.
    The returned batch is (x_p, x_ntc, p, c) with x_ntc.shape[0] == n_ntc.
    """

    def __init__(self, anndata, seed: int | None = None, combinatorial: bool = False, n_ntc: int = 20):
        rng = np.random.default_rng(seed)

        p_key = get_config('pert_key')
        NTC = get_config('ntc_label')
        treatment_key = get_config('treatment_key')

        pert_anndata = anndata[anndata.obs[p_key] != NTC]
        ntc_anndata = anndata[anndata.obs[p_key] == NTC]

        # ── metadata columns ────────────────────────────────────────────
        pertlbl      = pert_anndata.obs[p_key].values
        condlbl = pert_anndata.obs[treatment_key].values

        if pd.isna(pertlbl).any() or pd.isna(condlbl).any():
            raise ValueError(f"obs contains NA / NaN in {p_key} or {treatment_key}.")
        
        # unique conditions
        cond_unique = np.unique(condlbl)

        self.combinatorial = combinatorial
        pert_unique = np.unique(pertlbl)
        if not combinatorial:
            pert_unique = np.unique(pertlbl)
            self.perturbation_dict = {p: i for i, p in enumerate(pert_unique)}
        else:
            # combinatorial behavior: split on "+"
            parts = []
            for p in pertlbl:
                s = str(p)
                if "+" in s:
                    parts.extend([x.strip() for x in s.split("+") if x.strip()])
                else:
                    parts.append(s.strip())
            pert_unique = np.unique(np.array(parts, dtype=object))
            self.perturbation_dict = {p: i for i, p in enumerate(pert_unique)}

        self.condition_dict = {c: i for i, c in enumerate(cond_unique)}

        # if perturbation is on var, find indices
        self.perturbed_idx = np.where(anndata.var.index.isin(pert_unique))[0]
        self.perturbed_names = anndata.var.index[self.perturbed_idx].tolist()


        # ── expression matrix ───────────────────────────────────────────
        self.X_ntc = [ntc_anndata[ntc_anndata.obs[treatment_key] == cond].X.toarray() for cond in cond_unique]
        self.X_pert = pert_anndata.X.toarray() if hasattr(pert_anndata.X, "toarray") else pert_anndata.X

        self.P = pert_anndata.obs[p_key].values
        self.C = pert_anndata.obs[treatment_key].values
        self.C_ntc = ntc_anndata.obs[treatment_key].values

        # Convert C to indices
        self.C_indices = np.array([self.condition_dict[c] for c in self.C])
        self.C_ntc_indices = np.array([self.condition_dict[c] for c in self.C_ntc])
        # Convert P to indices
        if not combinatorial:
            self.P_indices = np.array([self.perturbation_dict[p] for p in self.P])
        else:
            # Nx2 indices: [p1, p2], with p2=-1 if single perturbation
            P2 = np.full((len(self.P), 2), -1, dtype=int)

            for i, p in enumerate(self.P):
                s = str(p).strip()
                if "+" in s:
                    toks = [x.strip() for x in s.split("+") if x.strip()]
                    # take first two if more than 2 are ever present
                    if len(toks) >= 1:
                        P2[i, 0] = self.perturbation_dict[toks[0]]
                    if len(toks) >= 2:
                        P2[i, 1] = self.perturbation_dict[toks[1]]
                else:
                    P2[i, 0] = self.perturbation_dict[s]

            self.P_indices = P2


        self.n_ntc = int(n_ntc)
        # Flat pool of all NTC cells for collate-time sampling
        self.X_ntc_flat = np.concatenate(self.X_ntc, axis=0).astype(np.float32)
        self.rng = rng

    # ── Dataset API ─────────────────────────────────────────────────────
    def __len__(self) -> int:
        return self.X_pert.shape[0]

    def __getitem__(self, idx: int):
        """Returns (x_p, p_idx, c_idx). Use get_collate_fn() with your DataLoader."""
        return self.X_pert[idx], self.P_indices[idx], self.C_indices[idx]

    def get_collate_fn(self):
        """
        Returns a collate_fn that samples n_ntc NTC cells *per condition* present
        in the batch (without replacement within each condition pool).

        Batch output: (x_p, x_ntc, p, c, c_ntc)
          x_p   : (n,                  G) float32
          x_ntc : (n_conds * n_ntc,    G) float32  — condition-grouped NTC cells
          p     : (n,) or (n, 2)          long
          c     : (n,)                    long      — condition of each perturbed cell
          c_ntc : (n_conds * n_ntc,)      long      — condition of each NTC cell
        """
        return _PerturbCollateFn(self.n_ntc, self.X_ntc)


class PerturbSimpleDataset(Dataset):
    """
    Flat dataset of (expression, perturbation_index) pairs for all cells.

    Unlike PerturbMatchingDataset, every cell — perturbed *and* NTC — is
    included.  NTC is assigned index 0; non-NTC perturbations follow in
    sorted order.  No on-the-fly NTC pairing is performed.

    This is the dataset used by cVAE/sVAE (and by the universal eval loop in
    PerturbModelBase.eval()) because all models can accept a plain (x, p_idx)
    batch.

    Parameters
    ----------
    anndata       : AnnData with obs[pert_key] containing perturbation labels.
    pert_key      : column in anndata.obs with perturbation labels.
    ntc_label     : label for non-targeting control cells.
    subset_obs    : optional boolean mask or integer indices to select a subset.
    combinatorial : if True, split labels on "+" and index individual genes.
                    NTC stays at 0; individual genes fill 1..P alphabetically.
                    P_indices becomes (N, 2) — NTC→[0,-1], single→[idx,-1],
                    combo→[idx1,idx2]. cVAE/sVAE models sum component shifts
                    for combinatorial cells. scGen leaves this False and treats
                    each combined label as a new atomic index.
    """

    def __init__(
        self,
        anndata,
        pert_key: str = "pert",
        ntc_label: str = "NTC",
        subset_obs=None,
        combinatorial: bool = False,
    ):
        if subset_obs is not None:
            anndata = anndata[subset_obs]

        self.pert_key = pert_key
        self.ntc_label = ntc_label
        self.combinatorial = combinatorial

        X = anndata.X
        self.X = (X.toarray() if hasattr(X, "toarray") else np.asarray(X)).astype(np.float32)

        raw_labels = anndata.obs[pert_key].values
        if pd.isna(raw_labels).any():
            raise ValueError(f"obs['{pert_key}'] contains NaN values.")

        if not combinatorial:
            unique_perts = np.unique(raw_labels)
            if ntc_label not in unique_perts:
                raise ValueError(
                    f"ntc_label '{ntc_label}' not found in obs['{pert_key}']. "
                    f"Available: {unique_perts[:10].tolist()} …"
                )
            # NTC → 0, everything else alphabetically
            non_ntc = sorted(p for p in unique_perts if p != ntc_label)
            self.perturbation_dict: dict[str, int] = {
                p: i for i, p in enumerate([ntc_label] + non_ntc)
            }
            self.P_indices = np.array(
                [self.perturbation_dict[p] for p in raw_labels], dtype=np.int64
            )
        else:
            # NTC stays at 0; individual gene components fill 1..P alphabetically.
            # This keeps the same NTC=0 convention as the non-combinatorial path so
            # model embedding tables (pert_emb padding_idx=0, action_prior_mean[0])
            # align without any index remapping.
            gene_set: set[str] = set()
            for lbl in raw_labels:
                s = str(lbl)
                if s == ntc_label:
                    continue
                if "+" in s:
                    gene_set.update(x.strip() for x in s.split("+") if x.strip())
                else:
                    gene_set.add(s.strip())
            gene_unique = sorted(gene_set)
            self.perturbation_dict = {ntc_label: 0}
            for i, g in enumerate(gene_unique, 1):
                self.perturbation_dict[g] = i

            # (N, 2): NTC→[0,-1], single gene→[idx,-1], combo→[idx1,idx2]
            P2 = np.full((len(raw_labels), 2), -1, dtype=np.int64)
            for i, lbl in enumerate(raw_labels):
                s = str(lbl)
                if s == ntc_label:
                    P2[i, 0] = 0
                elif "+" in s:
                    toks = [x.strip() for x in s.split("+") if x.strip()]
                    if len(toks) >= 1:
                        P2[i, 0] = self.perturbation_dict[toks[0]]
                    if len(toks) >= 2:
                        P2[i, 1] = self.perturbation_dict[toks[1]]
                else:
                    P2[i, 0] = self.perturbation_dict[s.strip()]
            self.P_indices = P2

        self.var_names: list[str] = anndata.var_names.tolist()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])
        if self.combinatorial:
            return x, torch.from_numpy(self.P_indices[idx])
        return x, torch.tensor(self.P_indices[idx], dtype=torch.long)

    @property
    def ntc_idx(self) -> int:
        return self.perturbation_dict[self.ntc_label]

    @property
    def n_perturbs(self) -> int:
        return len(self.perturbation_dict)

    @property
    def input_dim(self) -> int:
        return self.X.shape[1]

    def idx_to_pert(self) -> dict[int, str]:
        return {i: p for p, i in self.perturbation_dict.items()}