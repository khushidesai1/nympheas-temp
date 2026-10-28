# nympheas-qc

Fast, out-of-core cell and gene QC filtering plus normalization for
multi-sample spatial transcriptomics data stored as AnnData zarr stores (one
store per sample, local or on GCS).

This is the QC/filter/normalize part of Harjaisal's nympheas pipeline
(azizilab), pulled out so it can run on
other datasets. It keeps the techniques that made it fast on a 24-sample,
12.9M-cell CosMx colorectal cancer dataset. It drops the orchestrator,
provenance and GPU layers, which only served the later scVI and clustering
steps. It also adds the gene filter from the original scanpy crc pipeline.

## What it does

For a list of sample stores, `run_qc` / `nympheas-qc`:

1. **Cell filter, per sample.** Drops cells below the `cell_pctile`-th
   percentile of total counts (default 10), and always drops cells with zero
   counts (empty segmentation masks).
2. **Gene filter, shared across samples.** Drops genes below the
   `gene_pctile`-th percentile of total counts over the kept cells, pooled
   across all samples, so every sample keeps the same gene panel. The default
   of 0 keeps every gene.
3. **Normalize + log1p.** Scales each cell to `target_sum` (default: its
   sample's median total) and applies log1p. This matches
   `sc.pp.normalize_total` followed by `sc.pp.log1p`.
4. **Write.** Each sample becomes a complete AnnData zarr store next to its
   input (`processed.zarr` by default) containing:
   - normalized values in `X` and filtered raw counts in `layers/counts`;
   - the source `obs`, `var` and `obsm` subset to the kept cells and genes;
   - QC columns (`obs`: `total_counts`, `n_genes_by_counts`; `var`:
     `total_counts`, `n_cells_by_counts`);
   - the thresholds used, in `uns["qc"]`.

   Raw stores are never modified.

The filter rules are the old crc pipeline's
(`cell_min_transcript_counts_percentile`, `gene_min_counts_percentile`), with
one difference in scale: **percentiles here are 0–100**, and that config used
0–1 quantiles, so `0.1` there is `10` here.

## Why it's fast

The speed comes from how the data moves. These numbers were measured in
nympheas on the 24-sample dataset:

- **Lazy reads.** Counts are opened as dask arrays in 20k-cell chunks and
  never loaded whole.
- **Persist once.** The data is kept in cluster memory before several
  computations branch off it. This made filtering 12× faster, normalizing
  3.6× and writing 5.7×.
- **One batched `dask.compute` per pass.** All per-cell metrics for all
  samples, then all per-gene metrics. Four ~110 s passes became one ~117 s
  pass.
- **Small masks.** Masks are built in numpy from the per-cell and per-gene
  vectors and applied as concrete arrays, so dask knows output sizes without
  computing anything.
- **Single-threaded workers.** One worker per CPU, each with one thread.
  More threads per worker measured slower because of GIL contention.

## Install

```bash
pip install "nympheas-qc[gcs] @ git+https://github.com/khushidesai1/nympheas-temp.git"
```

Requires Python ≥ 3.12. The `gcs` extra adds `gcsfs` for `gs://` paths. For
development:

```bash
pip install -e ".[gcs,test]" && pytest
```

## Usage

Command line, for the proseg layout `<sample>/tables/proseg_table.zarr` with
gene symbols in `var["gene"]`:

```bash
nympheas-qc "gs://my-bucket/resegmented/*/tables/proseg_table.zarr" \
    --gene-key gene --cell-pctile 10 --gene-pctile 10 --summary-csv qc_summary.csv
```

Useful options for other datasets:

| option | use |
|---|---|
| `--counts-field layers/counts` | raw counts aren't in `X` |
| `--out-name '{stem}_qc.zarr'` | several stores share one folder (the default `processed.zarr` would collide, which is refused) |
| `--target-sum 1e4` | fixed normalization target instead of the per-sample median |
| `--no-write` | report thresholds and cell/gene counts only |
| `--n-workers`, `--memory-limit` | cluster size; defaults to one single-threaded worker per CPU |

From Python, inside your own dask cluster:

```python
from distributed import Client, LocalCluster
from nympheas_qc import run_qc

with LocalCluster(threads_per_worker=1) as cluster, Client(cluster):
    result = run_qc(stores, gene_key="gene", cell_pctile=10, gene_pctile=10)

result.summary       # per-sample cells/genes before and after, thresholds, targets
result.counts        # filtered raw counts per sample (dask, persisted)
result.normalized    # normalized + log1p per sample (dask, lazy)
```

Pass `out_name=None` to skip writing and keep working with the dask arrays.

## Validation

- **Tests.** `pytest` writes synthetic per-sample zarr stores and checks the
  output against the old crc pipeline's rules implemented with pandas and
  scanpy. The checks cover cell and gene masks, counts, normalized values,
  carried-over `obs`/`var`/`obsm`, QC columns, dense `X` and
  `layers/counts` inputs, and gene-panel and output-path errors.
- **Real data.** On one sample of the CosMx CRC dataset (541,690 cells ×
  6,717 genes, read from GCS), `cell_pctile=10` kept 487,948 cells. The
  filtered counts and normalized values were identical to nympheas's output
  for that sample. The run took 18 s on 16 workers.

## Limitations

- **Gene panels must match.** Every sample must list the same genes in the
  same order. This is checked up front.
- **Cluster memory.** The cluster holds the raw counts and the filtered copy
  at the same time. nympheas measured ~20 GB peak cluster memory for the
  persisted 24-sample counts on a 32-vCPU / 208 GB VM.
- **Cells emptied by the gene filter are kept.** The old pipeline did the
  same. They stay all-zero after normalization; their count is not reported.
- **Not carried over:** `varm`, `obsp`, `varp` and the source `uns`.
- **Writes go through the driver process.** Several samples are written in
  parallel threads (`--write-threads`).

## License

MIT, retaining the nympheas copyright notice (see `LICENSE`).
