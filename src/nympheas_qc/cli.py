"""Command line: `nympheas-qc STORE_OR_GLOB [...] [options]`."""

from __future__ import annotations

import argparse
import logging
import os

import fsspec
from distributed import Client, LocalCluster

from .pipeline import run_qc


def expand_stores(patterns: list[str]) -> list[str]:
    """Expand glob patterns (local or fsspec URLs) into store paths."""
    stores = []
    for pattern in patterns:
        if not any(c in pattern for c in "*?["):
            stores.append(pattern.rstrip("/"))
            continue
        fs, path = fsspec.core.url_to_fs(pattern)
        matches = sorted(fs.glob(path))
        if not matches:
            raise SystemExit(f"no stores match {pattern}")
        protocols = fs.protocol if isinstance(fs.protocol, tuple) else (fs.protocol,)
        local = "file" in protocols
        stores += [m if local else fs.unstrip_protocol(m) for m in matches]
    return stores


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="nympheas-qc",
        description="Filter and normalize AnnData zarr stores (one per sample) out of core.",
    )
    p.add_argument(
        "stores", nargs="+",
        help="sample stores, local or gs://...; quoted glob patterns are expanded",
    )
    p.add_argument("--counts-field", default="X", help="raw counts in each store, e.g. X or layers/counts")
    p.add_argument("--gene-key", help="var column with gene names (default: var index)")
    p.add_argument(
        "--cell-pctile", type=float, default=10.0,
        help="drop cells below this percentile (0-100) of total counts, per sample; "
        "zero-count cells are always dropped. The old crc config's 0.1 is 10 here",
    )
    p.add_argument(
        "--gene-pctile", type=float, default=0.0,
        help="drop genes below this percentile (0-100) of total counts pooled over all "
        "samples; 0 keeps every gene",
    )
    p.add_argument("--target-sum", type=float, help="normalize every cell to this total (default: sample median)")
    p.add_argument(
        "--out-name", default="processed.zarr",
        help="output store written next to each input; may use {stem}, e.g. '{stem}_qc.zarr'",
    )
    p.add_argument("--no-write", action="store_true", help="compute and report only")
    p.add_argument("--summary-csv", help="also write the per-sample summary here")
    p.add_argument("--chunk-rows", type=int, default=20_000)
    p.add_argument("--n-workers", type=int, default=os.cpu_count())
    p.add_argument("--memory-limit", default="auto", help="per worker, e.g. 6GB")
    p.add_argument("--write-threads", type=int, default=4)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    stores = expand_stores(args.stores)
    logging.info("%d stores", len(stores))

    # One single-threaded worker per CPU: nympheas measured more threads per
    # worker as slower (GIL contention), not faster, for this workload.
    with (
        LocalCluster(
            n_workers=args.n_workers, threads_per_worker=1, memory_limit=args.memory_limit
        ) as cluster,
        Client(cluster),
    ):
        result = run_qc(
            stores,
            counts_field=args.counts_field,
            gene_key=args.gene_key,
            cell_pctile=args.cell_pctile,
            gene_pctile=args.gene_pctile,
            target_sum=args.target_sum,
            out_name=None if args.no_write else args.out_name,
            chunk_rows=args.chunk_rows,
            write_threads=args.write_threads,
        )

    print(result.summary.to_string(index=False))
    if args.summary_csv:
        result.summary.to_csv(args.summary_csv, index=False)


if __name__ == "__main__":
    main()
