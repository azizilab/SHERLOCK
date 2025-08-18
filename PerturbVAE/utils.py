import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from pathlib import Path
from typing import List, Union, Literal, Sequence
import scanpy as sc
from typing import Sequence, Union, Literal, Tuple, List
from numpy.linalg import svd
from statsmodels.stats.multitest import multipletests
import pyro
from module import GeneModuleOutputLayer




def _build_incidence_matrix(module_dict, G, device):
    K = len(module_dict)
    H = torch.zeros(K, G, device=device)
    for k, genes in enumerate(module_dict.values()):
        H[k, torch.as_tensor(genes, device=device, dtype=torch.long)] = 1.0
    return H  # (K,G)

def _gene_weights_from_graph(Wp, H):
    s = (Wp.unsqueeze(0) @ H).squeeze(0) 
    s = torch.relu(s)
    if s.max() > 0:
        s = s / (s.max() + 1e-8)
    return s

def _gene_weights_from_pi(pi, dec, module_aware_dec=True):
    W1 = dec[0].weight
    if module_aware_dec:
        W2 = torch.nn.functional.softplus(dec[2].W_raw) * dec[2].M
        W2 = W2 / W2.norm(p=2, dim=0, keepdim=True).clamp_min(1e-6)
    else:
        W2 = dec[2].weight      
    print(W1.shape, W2.shape)
    # project to gene space   
    U = (W2.detach()**2) @ (W1.detach()**2) 
    print(U.shape, pi.shape)
    s = (U @ pi.clamp(0,1))                   # (G,)
    s = torch.relu(s)
    s = s / (s.sum() + 1e-8)
    return s