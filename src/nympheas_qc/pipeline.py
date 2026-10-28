"""Multi-sample QC: cell filter, shared gene filter, normalize + log1p, write.

The speed comes from how data moves, not from the filter rules: counts are
read lazily in row chunks, persisted once in cluster memory before several
computations branch off them, every metric is computed in one batched
`dask.compute`, and masks are built in numpy from the small per-cell and
per-gene vectors. These are the measured lessons from nympheas
(`drafts/PIPELINE_NOTES.md`): persisting first made filtering 12x,
normalizing 3.6x and writing 5.7x faster, and batching four QC metrics into
one compute took four ~110s passes down to one ~117s pass.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import dask
import dask.array as da
import numpy as np
import pandas as pd
from distributed import futures_of, wait

from . import io, metrics

log = logging.getLogger(__name__)


@dataclass
class QCResult:
    """Per-sample results, in the order of the input stores. `counts` are the
    filtered raw counts (persisted when a distributed client is active);
    `normalized` are lazy graphs over them."""

    summary: pd.DataFrame
    cell_masks: list[np.ndarray]
    gene_mask: np.ndarray
    gene_threshold: float
    counts: list[da.Array]
    normalized: list[da.Array]


def _persist(arrays: list[da.Array]) -> list[da.Array]:
    """Persist all arrays in one call and block until they are in memory."""
    arrays = dask.persist(*arrays)
    if futures := futures_of(arrays):
        wait(futures)
    return list(arrays)


def _split(values: tuple, n: int) -> tuple[list, list]:
    return list(values[:n]), list(values[n:])


def run_qc(
    stores: list[str],
    *,
    counts_field: str = "X",
    gene_key: str | None = None,
    cell_pctile: float = 10.0,
    gene_pctile: float = 0.0,
    target_sum: float | None = None,
    out_name: str | None = "processed.zarr",
    chunk_rows: int = 20_000,
    write_threads: int = 4,
) -> QCResult:
    """Filter, normalize and (unless `out_name` is None) write every sample.

    1. Drop cells below the `cell_pctile`-th percentile of total counts,
       per sample, plus every zero-count cell.
    2. Drop genes below the `gene_pctile`-th percentile of total counts over
       the kept cells, pooled across all samples, so every sample keeps the
       same gene panel. `gene_pctile=0` keeps every gene.
    3. Scale each cell to `target_sum` (default: its sample's median total)
       and log1p, like `sc.pp.normalize_total` + `sc.pp.log1p`.
    4. Write each sample to `out_name` next to its input store (see
       `io.output_store`) as an AnnData zarr store.

    Percentiles are on a 0-100 scale. Run inside a `distributed.Client` for
    the parallel, persisted path; without one, dask's threaded scheduler is
    used and everything is held in local memory.
    """
    if not stores:
        raise ValueError("no input stores given")
    n = len(stores)
    outputs = io.output_stores(stores, out_name) if out_name else []
    io.check_gene_panels(stores, gene_key)
    t0 = time.perf_counter()

    X = _persist([io.open_counts(s, counts_field, chunk_rows) for s in stores])
    log.info("read + persist %d samples: %.1fs", n, time.perf_counter() - t0)

    # Pass 1: per-cell metrics for every sample in one compute, so dask reads
    # each chunk once instead of once per metric per sample.
    cell_totals, cell_nnz = _split(
        dask.compute(*map(metrics.row_totals, X), *map(metrics.row_nnz, X)), n
    )
    cell_masks, cell_thresholds = zip(*(metrics.cell_mask(t, cell_pctile) for t in cell_totals))
    # Indexing with a concrete numpy mask lets dask derive the output chunk
    # sizes without computing anything.
    X = [metrics.ensure_csr(x[m]) for x, m in zip(X, cell_masks)]

    # Pass 2: per-gene metrics over the kept cells, again in one compute.
    gene_totals, gene_nnz = _split(
        dask.compute(*map(metrics.col_totals, X), *map(metrics.col_nnz, X)), n
    )
    gene_mask, gene_threshold = metrics.gene_mask(np.sum(gene_totals, axis=0), gene_pctile)
    if not gene_mask.all():
        X = [metrics.ensure_csr(x[:, gene_mask]) for x in X]
    # Rebinding X drops the last reference to the unfiltered persisted copy.
    X = _persist(X)
    log.info(
        "QC + filter: %.1fs; kept %d/%d genes",
        time.perf_counter() - t0, gene_mask.sum(), gene_mask.size,
    )

    # Totals for normalization: reuse pass 1 unless genes were dropped.
    if gene_mask.all():
        norm_totals = [t[m] for t, m in zip(cell_totals, cell_masks)]
    else:
        norm_totals = list(dask.compute(*map(metrics.row_totals, X)))
    targets = [
        float(target_sum) if target_sum is not None else metrics.normalization_target(t)
        for t in norm_totals
    ]
    normalized = [metrics.normalize_log1p(x, t, g) for x, t, g in zip(X, norm_totals, targets)]

    summary = pd.DataFrame(
        {
            "store": stores,
            "n_cells_before": [m.size for m in cell_masks],
            "n_cells_after": [int(m.sum()) for m in cell_masks],
            "cell_total_counts_threshold": cell_thresholds,
            "median_total_counts": [float(np.median(t)) for t in cell_totals],
            "n_genes_before": gene_mask.size,
            "n_genes_after": int(gene_mask.sum()),
            "normalize_target": targets,
            "output": outputs or None,
        }
    )

    if out_name:

        def write(i: int) -> None:
            m = cell_masks[i]
            io.write_processed(
                outputs[i],
                stores[i],
                counts=X[i],
                normalized=normalized[i],
                cell_mask=m,
                gene_mask=gene_mask,
                obs_columns={
                    "total_counts": cell_totals[i][m],
                    "n_genes_by_counts": cell_nnz[i][m],
                },
                var_columns={
                    "total_counts": gene_totals[i][gene_mask],
                    "n_cells_by_counts": gene_nnz[i][gene_mask],
                },
                uns={
                    "qc": {
                        "source": stores[i],
                        "counts_field": counts_field,
                        "cell_pctile": float(cell_pctile),
                        "cell_total_counts_threshold": cell_thresholds[i],
                        "gene_pctile": float(gene_pctile),
                        "gene_total_counts_threshold": gene_threshold,
                        "normalize_target": targets[i],
                    }
                },
            )
            log.info("wrote %s", outputs[i])

        t1 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, min(write_threads, n))) as pool:
            list(pool.map(write, range(n)))
        log.info("write %d samples: %.1fs", n, time.perf_counter() - t1)

    return QCResult(
        summary=summary,
        cell_masks=list(cell_masks),
        gene_mask=gene_mask,
        gene_threshold=gene_threshold,
        counts=X,
        normalized=normalized,
    )
