"""Reading per-sample AnnData zarr stores and writing the processed ones.

Paths are plain strings, local or fsspec URLs (`gs://...`).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import zarr

from . import metrics


def output_store(store: str, out_name: str) -> str:
    """Path of the processed store written next to `store`. `out_name` may
    use `{stem}`, the input store's name without `.zarr`."""
    parent, _, name = store.rstrip("/").rpartition("/")
    out = out_name.format(stem=name.removesuffix(".zarr"))
    return f"{parent}/{out}" if parent else out


def output_stores(stores: list[str], out_name: str) -> list[str]:
    """Output path per input store; refuses collisions and in-place writes,
    which would silently overwrite one sample with another or the raw data."""
    outputs = [output_store(s, out_name) for s in stores]
    seen: dict[str, str] = {}
    for store, out in zip(stores, outputs):
        if out.rstrip("/") == store.rstrip("/"):
            raise ValueError(f"output {out} would overwrite its input store")
        if out in seen:
            raise ValueError(
                f"{seen[out]} and {store} would both write to {out}; "
                "pass an out_name using {stem}, e.g. '{stem}_qc.zarr'"
            )
        seen[out] = store
    return outputs


def open_counts(store: str, field: str = "X", chunk_rows: int = 20_000) -> da.Array:
    """Lazy counts matrix of one store, chunked by `chunk_rows` cells and the
    full gene axis, with CSR chunks. Nothing is read until computed."""
    elem = zarr.open_group(store, mode="r")[field]
    shape = elem.attrs["shape"] if isinstance(elem, zarr.Group) else elem.shape
    X = ad.experimental.read_elem_lazy(elem, chunks=(chunk_rows, shape[1]))
    return metrics.ensure_csr(X)


def _gene_names(store: str, gene_key: str | None) -> np.ndarray:
    var = ad.io.read_elem(zarr.open_group(store, mode="r")["var"])
    names = var.index if gene_key is None else var[gene_key]
    return np.asarray(names).astype(str)


def check_gene_panels(stores: list[str], gene_key: str | None = None) -> np.ndarray:
    """Return the shared gene panel, or raise if any store's genes differ in
    content or order. A shared gene mask is only meaningful if they match.
    `gene_key` is the `var` column holding gene names (default: the index;
    proseg tables keep a numeric index and the names in `var["gene"]`)."""
    with ThreadPoolExecutor(max_workers=min(8, len(stores))) as pool:
        panels = list(pool.map(lambda s: _gene_names(s, gene_key), stores))
    for store, panel in zip(stores[1:], panels[1:]):
        if not np.array_equal(panel, panels[0]):
            raise ValueError(
                f"gene panel of {store} differs from {stores[0]}; "
                "every sample must list the same genes in the same order"
            )
    return panels[0]


def _subset_rows(value, mask: np.ndarray, index: pd.Index):
    if isinstance(value, pd.DataFrame):
        return value.iloc[mask].set_axis(index)
    return value[mask]


def write_processed(
    out_store: str,
    src_store: str,
    *,
    counts: da.Array,
    normalized: da.Array,
    cell_mask: np.ndarray,
    gene_mask: np.ndarray,
    obs_columns: dict[str, np.ndarray],
    var_columns: dict[str, np.ndarray],
    uns: dict,
) -> None:
    """Write one sample as a complete AnnData zarr store: normalized values in
    `X`, filtered raw counts in `layers/counts`, and the source store's
    `obs`/`var`/`obsm` subset to the kept cells and genes (plus QC columns).
    `varm`, `obsp`, `varp` and the source `uns` are not carried over."""
    src = zarr.open_group(src_store, mode="r")
    obs = ad.io.read_elem(src["obs"]).iloc[cell_mask].assign(**obs_columns)
    var = ad.io.read_elem(src["var"]).iloc[gene_mask].assign(**var_columns)
    obsm = {
        key: _subset_rows(ad.io.read_elem(src["obsm"][key]), cell_mask, obs.index)
        for key in (src["obsm"].keys() if "obsm" in src else [])
    }
    adata = ad.AnnData(
        X=normalized, layers={"counts": counts}, obs=obs, var=var, obsm=obsm, uns=uns
    )
    adata.write_zarr(out_store)
