from __future__ import annotations

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from nympheas_qc import metrics

from conftest import make_counts


def test_cell_mask_matches_old_crc_pipeline_rule():
    # crc_cosmx preprocess.py: obs["transcript_counts"] >= quantile(0.1)
    totals = np.random.default_rng(0).poisson(40, 500).astype(float)
    totals[[3, 77]] = 0
    mask, threshold = metrics.cell_mask(totals, 10)
    s = pd.Series(totals)
    np.testing.assert_array_equal(mask, (s >= s.quantile(0.1)).to_numpy() & (totals > 0))
    assert threshold == s.quantile(0.1)


def test_zero_count_cells_dropped_even_at_pctile_zero():
    totals = np.array([0.0, 5, 0, 9, 12, 0])
    mask, threshold = metrics.cell_mask(totals, 0)
    assert threshold == 0
    np.testing.assert_array_equal(mask, totals > 0)


def test_gene_mask_matches_old_crc_pipeline_rule():
    # crc_cosmx preprocess.py: var["total_counts"] >= quantile(0.1)
    totals = np.random.default_rng(1).gamma(0.8, 300, 200)
    mask, _ = metrics.gene_mask(totals, 10)
    s = pd.Series(totals)
    np.testing.assert_array_equal(mask, (s >= s.quantile(0.1)).to_numpy())


def test_gene_mask_pctile_zero_keeps_every_gene():
    totals = np.array([0.0, 0, 3, 10])
    mask, _ = metrics.gene_mask(totals, 0)
    assert mask.all()


@pytest.mark.parametrize("dense", [False, True])
def test_lazy_metrics_match_numpy(client, dense):
    X = make_counts(230, 17, seed=2, zero_cells=(0, 100))
    data = da.from_array(X.toarray() if dense else X, chunks=(37, 17))
    lazy = metrics.ensure_csr(data)
    assert lazy._meta.format == "csr"

    dense_X = X.toarray().astype(np.float64)
    totals, nnz, gtotals, gnnz = da.compute(
        metrics.row_totals(lazy),
        metrics.row_nnz(lazy),
        metrics.col_totals(lazy),
        metrics.col_nnz(lazy),
    )
    np.testing.assert_allclose(totals, dense_X.sum(axis=1))
    np.testing.assert_array_equal(nnz, (dense_X != 0).sum(axis=1))
    np.testing.assert_allclose(gtotals, dense_X.sum(axis=0))
    np.testing.assert_array_equal(gnnz, (dense_X != 0).sum(axis=0))


@pytest.mark.parametrize("target_sum", [None, 1e4])
def test_normalize_matches_scanpy(client, target_sum):
    sc = pytest.importorskip("scanpy")
    # Enough empty cells that a median including them would differ from
    # scanpy's, which ignores them.
    X = make_counts(150, 12, seed=3, zero_cells=tuple(range(5, 150, 3)))
    totals = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    target = target_sum if target_sum is not None else metrics.normalization_target(totals)

    out = metrics.normalize_log1p(da.from_array(X, chunks=(40, 12)), totals, target).compute()

    ref = ad.AnnData(X.astype(np.float64))
    sc.pp.normalize_total(ref, target_sum=target_sum)
    sc.pp.log1p(ref)
    assert sp.issparse(out)
    np.testing.assert_allclose(out.toarray(), ref.X.toarray(), rtol=1e-10, atol=1e-12)
    assert np.all(out.toarray()[5] == 0)
