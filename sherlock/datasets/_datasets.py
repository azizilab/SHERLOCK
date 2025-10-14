import scanpy as sc
import pandas as pd
from pathlib import Path

from ..preprocessing import slk_prepare_data

HERE = Path(__file__).parent

def replogle():
    data = sc.read_h5ad(HERE / 'replogle.h5ad.gzip')

    slk_prepare_data(data, pert_key='gene', ntc_label='non-targeting', treatment_key=None, inplace=True)

    return data