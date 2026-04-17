import scgen
import scanpy as sc

def train_scgen(adata, pert_col, control_label, n_epochs=100):
    adata = adata.copy()
    adata.obs_names_make_unique()

    adata.obs["condition"] = adata.obs[pert_col].astype(str)
    adata.obs["cell_type"] = "all_cells"

    scgen.SCGEN.setup_anndata(
        adata,
        batch_key="condition",
        labels_key="cell_type",
    )

    model = scgen.SCGEN(adata)
    model.train(max_epochs=n_epochs)

    # keep the exact trained AnnData attached
    model._sherlock_adata = adata
    return model