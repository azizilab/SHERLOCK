from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from .._configs import get_config
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from .._configs import get_config

class PerturbBenchmarkDataset(Dataset):
    """
    Generic benchmark dataset for perturbation models.

    Returns:
      x : float32 expression
      p : int64 perturbation index (includes NTC)
    """
    def __init__(self, anndata):
    
        p_key = get_config("pert_key")
        ntc_label = get_config("ntc_label")

        X = anndata.X.toarray() if hasattr(anndata.X, "toarray") else anndata.X
        self.X = np.asarray(X, dtype=np.float32)

        # Perturbation labels
        if p_key not in anndata.obs:
            raise KeyError(f"AnnData.obs missing '{p_key}'")

        self.P_names = anndata.obs[p_key].astype(str).values

        if pd.isna(self.P_names).any():
            raise ValueError(f"obs['{p_key}'] contains NaN")

        pert_unique = np.unique(self.P_names)

        if ntc_label not in pert_unique:
            raise ValueError(
                f"NTC label '{ntc_label}' not found in obs['{p_key}']. "
                "ATE cannot be computed."
            )

        self.perturbation_dict = {p: i for i, p in enumerate(pert_unique)}
        self.idx_to_pert = {i: p for p, i in self.perturbation_dict.items()}
        self.ntc_idx = self.perturbation_dict[ntc_label]

        self.P_indices = np.array(
            [self.perturbation_dict[p] for p in self.P_names],
            dtype=np.int64,
        )
        self.is_ntc = (self.P_indices == self.ntc_idx)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.P_indices[idx]