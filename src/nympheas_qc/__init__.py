"""Out-of-core QC filtering and normalization for multi-sample AnnData zarr stores."""

from .pipeline import QCResult, run_qc

__all__ = ["QCResult", "run_qc"]
