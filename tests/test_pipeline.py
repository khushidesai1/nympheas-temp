"""End-to-end: synthetic per-sample zarr stores in, processed stores out,
checked against the old crc pipeline's rules written with pandas/scanpy."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from nympheas_qc import run_qc
from nympheas_qc.io import output_store

from conftest import make_counts, sample_path, write_store

SIZES = [137, 260, 95]
N_GENES = 40


@pytest.fixture
def samples(tmp_path):
    counts, stores = [], []
    for i, n in enumerate(SIZES):
        X = make_counts(n, N_GENES, seed=10 + i, zero_cells=(0, n // 2))
        stores.append(write_store(sample_path(tmp_path, f"S{i}"), X, seed=20 + i))
        counts.append(X)
    return stores, counts


def reference(counts, cell_pctile, gene_pctile, target_sum):
    """Cell filter per sample, then the gene filter on totals pooled over the
    kept cells of all samples, then scanpy normalize_total + log1p."""
    cell_masks, kept = [], []
    for X in counts:
        totals = pd.Series(np.asarray(X.sum(axis=1)).ravel())
        keep = (totals >= totals.quantile(cell_pctile / 100)).to_numpy() & (totals > 0).to_numpy()
        cell_masks.append(keep)
        kept.append(X[keep])
    pooled = pd.Series(sum(np.asarray(K.sum(axis=0), dtype=np.float64).ravel() for K in kept))
    gene_keep = (pooled >= pooled.quantile(gene_pctile / 100)).to_numpy()

    sc = pytest.importorskip("scanpy")
    refs = []
    for K in kept:
        a = ad.AnnData(K[:, gene_keep].astype(np.float64))
        a.layers["counts"] = K[:, gene_keep]
        sc.pp.normalize_total(a, target_sum=target_sum)
        sc.pp.log1p(a)
        refs.append(a)
    return cell_masks, gene_keep, refs


@pytest.mark.parametrize(
    ("gene_pctile", "target_sum"), [(0.0, None), (25.0, None), (25.0, 1e4)]
)
def test_matches_reference(client, samples, gene_pctile, target_sum):
    stores, counts = samples
    result = run_qc(
        stores,
        gene_key="gene",
        cell_pctile=10.0,
        gene_pctile=gene_pctile,
        target_sum=target_sum,
        chunk_rows=50,
    )
    cell_masks, gene_keep, refs = reference(counts, 10.0, gene_pctile, target_sum)

    np.testing.assert_array_equal(result.gene_mask, gene_keep)
    if gene_pctile > 0:
        assert 0 < gene_keep.sum() < N_GENES

    for i, (store, X, m, ref) in enumerate(zip(stores, counts, cell_masks, refs)):
        np.testing.assert_array_equal(result.cell_masks[i], m)
        src = ad.read_zarr(store)
        out = ad.read_zarr(output_store(store, "processed.zarr"))

        assert out.shape == ref.shape
        np.testing.assert_array_equal(out.layers["counts"].toarray(), ref.layers["counts"].toarray())
        np.testing.assert_allclose(out.X.toarray(), ref.X.toarray(), rtol=1e-10, atol=1e-12)

        assert list(out.obs_names) == list(src.obs_names[m])
        assert list(out.obs["fov"]) == list(src.obs["fov"][m])
        np.testing.assert_array_equal(out.obsm["spatial"], src.obsm["spatial"][m])
        assert list(out.var["gene"]) == list(src.var["gene"][gene_keep])

        dense = X.toarray().astype(np.float64)
        np.testing.assert_allclose(out.obs["total_counts"], dense.sum(axis=1)[m])
        np.testing.assert_array_equal(out.obs["n_genes_by_counts"], (dense != 0).sum(axis=1)[m])
        np.testing.assert_allclose(out.var["total_counts"], dense[m].sum(axis=0)[gene_keep])
        assert out.uns["qc"]["gene_pctile"] == gene_pctile

    assert list(result.summary["n_cells_after"]) == [int(m.sum()) for m in cell_masks]


def test_dense_x_and_counts_layer_inputs(client, tmp_path):
    """Dense X, and counts stored in layers/counts instead of X, give the same
    result as sparse counts in X."""
    X = make_counts(120, N_GENES, seed=30, zero_cells=(3,))
    sparse_store = write_store(sample_path(tmp_path / "a", "S0"), X, seed=1)
    dense_store = write_store(sample_path(tmp_path / "b", "S0"), X, seed=1, dense=True)
    layer_store = write_store(sample_path(tmp_path / "c", "S0"), X, seed=1, counts_layer=True)

    base = run_qc([sparse_store], gene_pctile=20, out_name=None, chunk_rows=50)
    for store, field in [(dense_store, "X"), (layer_store, "layers/counts")]:
        other = run_qc([store], counts_field=field, gene_pctile=20, out_name=None, chunk_rows=50)
        np.testing.assert_array_equal(other.counts[0].compute().toarray(), base.counts[0].compute().toarray())
        np.testing.assert_allclose(
            other.normalized[0].compute().toarray(), base.normalized[0].compute().toarray()
        )


def test_no_write_leaves_disk_untouched(client, samples, tmp_path):
    stores, _ = samples
    result = run_qc(stores, out_name=None, chunk_rows=50)
    assert not list(tmp_path.rglob("processed.zarr"))
    assert all(sp.issparse(n.compute()) for n in result.normalized)


def test_mismatched_gene_panels_raise(tmp_path):
    X = make_counts(30, 5, seed=4)
    a = write_store(sample_path(tmp_path, "A"), X, seed=1, gene_names=list("ABCDE"))
    b = write_store(sample_path(tmp_path, "B"), X, seed=1, gene_names=list("ABCED"))
    with pytest.raises(ValueError, match="gene panel"):
        run_qc([a, b], gene_key="gene", out_name=None)


def test_colliding_outputs_raise(tmp_path):
    X = make_counts(30, 5, seed=5)
    a = write_store(tmp_path / "one.zarr", X, seed=1)
    b = write_store(tmp_path / "two.zarr", X, seed=1)
    with pytest.raises(ValueError, match="both write"):
        run_qc([a, b])


def test_output_store_paths():
    store = "gs://bucket/resegmented/S1/tables/proseg_table.zarr"
    assert output_store(store, "processed.zarr") == "gs://bucket/resegmented/S1/tables/processed.zarr"
    assert output_store(store + "/", "{stem}_qc.zarr") == (
        "gs://bucket/resegmented/S1/tables/proseg_table_qc.zarr"
    )
