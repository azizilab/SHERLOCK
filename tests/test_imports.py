"""Smoke tests: the package imports and exposes its public namespaces."""


def test_import_namespaces():
    import sherlock as slk

    assert isinstance(slk.__version__, str)
    for ns in ("tl", "pp", "pl", "configs", "datasets"):
        assert hasattr(slk, ns)


def test_public_api():
    import sherlock as slk

    for name in ("run", "eval", "save_results", "load_results", "VAE", "cluster_rho",
                 "ev_sig", "compute_clustering_metrics", "regress_gi_params", "classify_gi"):
        assert hasattr(slk.tl, name), name
    for name in ("treat_effect", "slk_prepare_data"):
        assert hasattr(slk.pp, name), name
    for name in ("plot_r2", "plot_corr", "plot_zcorr", "plot_ev", "plot_ev_sig",
                 "plot_cf_bipartite", "plot_cf_target_overlap"):
        assert hasattr(slk.pl, name), name
