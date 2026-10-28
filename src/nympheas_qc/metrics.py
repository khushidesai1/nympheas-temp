"""Per-chunk QC metrics, filter masks, and the normalize + log1p transform.

Everything here either runs on one row-chunk of the counts matrix (a scipy
CSR block) or is plain numpy over a small per-cell / per-gene vector, so the
full matrix is never materialized in one place.
"""

from __future__ import annotations

import dask.array as da
import numpy as np
import scipy.sparse as sp

_F64 = np.float64


def to_csr(block):
    """Coerce one chunk (dense, CSC or CSR) to CSR."""
    if sp.issparse(block):
        return block if block.format == "csr" else block.tocsr()
    return sp.csr_matrix(block)


def ensure_csr(X: da.Array) -> da.Array:
    """Lazily make every chunk of `X` CSR and record that in its meta.

    Besides giving the rest of the package one layout to handle, this keeps
    writes on anndata's sparse dask writer: its dense dask writer hardcodes
    the threaded scheduler and fails on data persisted in a distributed
    cluster (see nympheas `drafts/PIPELINE_NOTES.md`)."""
    if getattr(X._meta, "format", None) == "csr":
        return X
    return X.map_blocks(to_csr, dtype=X.dtype, meta=sp.csr_matrix((0, 0), dtype=X.dtype))


def _row_sum(block):
    return np.asarray(block.sum(axis=1, dtype=_F64)).ravel()


def _row_nnz(block):
    return np.asarray((block != 0).sum(axis=1)).ravel().astype(np.int64)


def _col_sum(block):
    return np.asarray(block.sum(axis=0, dtype=_F64)).reshape(1, -1)


def _col_nnz(block):
    return np.asarray((block != 0).sum(axis=0)).reshape(1, -1).astype(np.int64)


def _full_width(X: da.Array) -> da.Array:
    return X if X.numblocks[1] == 1 else X.rechunk({1: -1})


def _per_row(X: da.Array, fn, dtype) -> da.Array:
    X = _full_width(X)
    return X.map_blocks(fn, dtype=dtype, drop_axis=1, meta=np.empty(0, dtype=dtype))


def _per_col(X: da.Array, fn, dtype) -> da.Array:
    """One `(1, n_genes)` partial per row-chunk, then summed over chunks."""
    X = _full_width(X)
    partial = X.map_blocks(
        fn,
        dtype=dtype,
        chunks=((1,) * X.numblocks[0], X.chunks[1]),
        meta=np.empty((0, 0), dtype=dtype),
    )
    return partial.sum(axis=0)


def row_totals(X: da.Array) -> da.Array:
    """Lazy total counts per cell, shape `(n_cells,)`."""
    return _per_row(X, _row_sum, _F64)


def row_nnz(X: da.Array) -> da.Array:
    """Lazy number of detected genes per cell, shape `(n_cells,)`."""
    return _per_row(X, _row_nnz, np.int64)


def col_totals(X: da.Array) -> da.Array:
    """Lazy total counts per gene, shape `(n_genes,)`."""
    return _per_col(X, _col_sum, _F64)


def col_nnz(X: da.Array) -> da.Array:
    """Lazy number of cells detecting each gene, shape `(n_genes,)`."""
    return _per_col(X, _col_nnz, np.int64)


def cell_mask(totals: np.ndarray, pctile: float) -> tuple[np.ndarray, float]:
    """Keep cells at or above the `pctile`-th percentile (0-100 scale) of
    total counts, and always drop cells with zero counts.

    Same rule as the old crc pipeline's `cell_min_transcript_counts_percentile`,
    which used a 0-1 quantile scale (0.1 there is 10 here). The zero guard
    matters at `pctile=0`, where the threshold is the minimum (often 0) and
    would otherwise keep empty segmentation-background "cells"; left in, those
    were measured to collapse into one spurious cell type downstream."""
    threshold = float(np.percentile(totals, pctile))
    return (totals >= threshold) & (totals > 0), threshold


def gene_mask(totals: np.ndarray, pctile: float) -> tuple[np.ndarray, float]:
    """Keep genes at or above the `pctile`-th percentile (0-100 scale) of
    total counts.

    Same rule as the old crc pipeline's `gene_min_counts_percentile` (0-1
    scale there). At `pctile=0` the threshold is the minimum, so every gene
    is kept."""
    threshold = float(np.percentile(totals, pctile))
    return totals >= threshold, threshold


def normalization_target(totals: np.ndarray) -> float:
    """Median total over cells with nonzero counts, matching
    `sc.pp.normalize_total(target_sum=None)`."""
    positive = totals[totals > 0]
    return float(np.median(positive)) if positive.size else 1.0


def normalize_log1p_block(block, totals, target: float):
    """Scale each row of one CSR chunk to sum to `target`, then log1p.
    Rows with a zero total stay all-zero."""
    totals = np.asarray(totals, dtype=_F64).ravel()
    scale = target / np.where(totals == 0, 1.0, totals)
    out = (sp.diags(scale) @ block).tocsr().astype(_F64)
    out.data = np.log1p(out.data)
    return out


def normalize_log1p(X: da.Array, totals: np.ndarray, target: float) -> da.Array:
    """Lazy `normalize_log1p_block` over every chunk of `X`. `totals` is the
    already-computed per-cell totals vector, rewrapped to `X`'s row chunks so
    no second row-sum pass is needed."""
    totals_da = da.from_array(np.asarray(totals, dtype=_F64), chunks=(X.chunks[0],))
    return X.map_blocks(
        normalize_log1p_block,
        totals_da[:, None],
        target,
        dtype=_F64,
        meta=sp.csr_matrix((0, 0), dtype=_F64),
    )
