"""Synthetic per-sample AnnData zarr stores and a small distributed cluster."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from distributed import Client, LocalCluster


def make_counts(
    n_cells: int,
    n_genes: int,
    *,
    seed: int,
    zero_cells: tuple[int, ...] = (),
    density: float = 0.4,
) -> sp.csr_matrix:
    """Poisson counts with per-gene rates spread over two orders of
    magnitude (so a gene percentile filter has low-count genes to drop),
    plus rows forced to zero to stand in for segmentation background."""
    rng = np.random.default_rng(seed)
    rates = rng.gamma(0.8, 3.0, size=n_genes)
    X = rng.poisson(rates, size=(n_cells, n_genes)).astype(np.float32)
    X[rng.random(X.shape) > density] = 0
    X[list(zero_cells)] = 0
    return sp.csr_matrix(X)


def write_store(
    path: Path,
    X: sp.csr_matrix,
    *,
    seed: int,
    dense: bool = False,
    counts_layer: bool = False,
    gene_names: list[str] | None = None,
) -> str:
    """Write one sample as an AnnData zarr store laid out like the proseg
    tables: numeric `var` index with gene symbols in `var["gene"]`, a
    categorical obs column, and `obsm["spatial"]`. With `counts_layer`, the
    counts go to `layers/counts` and `X` holds something else."""
    rng = np.random.default_rng(seed)
    n_cells, n_genes = X.shape
    obs = pd.DataFrame(
        {"fov": pd.Categorical(rng.integers(0, 4, n_cells).astype(str))},
        index=[f"{path.parent.parent.name}-cell{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        {"gene": gene_names or [f"GENE{j}" for j in range(n_genes)]},
        index=[str(j) for j in range(n_genes)],
    )
    counts = X.toarray() if dense else X
    adata = ad.AnnData(
        X=np.log1p(X.toarray()) if counts_layer else counts,
        obs=obs,
        var=var,
        obsm={"spatial": rng.random((n_cells, 2))},
    )
    if counts_layer:
        adata.layers["counts"] = counts
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_zarr(path)
    return str(path)


def sample_path(root: Path, name: str) -> Path:
    return root / name / "tables" / "proseg_table.zarr"


@pytest.fixture(scope="session")
def client():
    with (
        LocalCluster(
            n_workers=2, threads_per_worker=1, memory_limit="2GB", dashboard_address=":0"
        ) as cluster,
        Client(cluster) as c,
    ):
        yield c
