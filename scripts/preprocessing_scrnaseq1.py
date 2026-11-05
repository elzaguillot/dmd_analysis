"""scRNAseq1 (10x, healthy and DMD cells, day 5 to 14) preprocessing: QC, gene filtering, normalisation.

Starts from the two pools after CellRanger (forced cell number), HTO assignment and velocyto merge,
combined in one file (1-all_Velocyto.h5ad, 10,965 cells). Cells were selected upstream; this script
computes QC metrics, removes ribosomal genes and MALAT1, then filters genes and normalises the spliced
counts with scvelo, as in the original pipeline (output identical to 3-all_Velocyto.h5ad).

Usage:
    python scripts/preprocessing_scrnaseq1.py --input data/scrnaseq/1-all_Velocyto.h5ad --out results/scrnaseq1
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv

REPO_DIR = Path(__file__).resolve().parent.parent

# Sequencing pools (obs "batch") and the cell selection thresholds applied on the first CellRanger run,
# before CellRanger was rerun with a forced number of cells; drawn on the QC plots for reference only
POOLS = {"0": "pool 1 (day 5-8)", "1": "pool 2 (day 9-14)"}
POOL_THRESHOLDS = {"0": {"log10_umi": 3.9, "genes": 3800}, "1": {"log10_umi": 3.4, "genes": 1500}}
# Ribosomal genes are matched by prefix as in the original pipeline, which also removes a few
# non-ribosomal genes (RPS6K* kinases, RPS19BP1, antisense transcripts such as RPL34-AS1)
RIBO_PREFIXES = ("RPS", "RPL", "MRPS", "MRPL")
OTHER_REMOVED_GENES = ["MALAT1"]
MIN_SHARED_COUNTS = 1  # gene kept if detected spliced and unspliced in the same cell at least once


def quality_metrics(adata: ad.AnnData):
    """UMI, detected genes, % mitochondrial and ribosomal counts per cell, on the CellRanger counts."""
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.upper().str.startswith(RIBO_PREFIXES)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo"], layer="matrix", log1p=False, inplace=True)


def plot_quality(adata: ad.AnnData, fig_dir: Path):
    """Per pool: knee plot, UMI and detected genes distributions, % mitochondrial and ribosomal counts."""
    fig, axes = plt.subplots(len(POOLS), 5, figsize=(22, 4 * len(POOLS)), squeeze=False)
    for row, (pool, label) in zip(axes, POOLS.items()):
        obs = adata.obs[adata.obs["batch"] == pool]
        thresholds = POOL_THRESHOLDS[pool]
        umi = np.sort(obs["total_counts"].to_numpy())[::-1]
        row[0].loglog(np.arange(1, len(umi) + 1), umi)
        row[0].set_xlabel("cell rank")
        row[0].set_ylabel("UMI")
        row[1].hist(np.log10(obs["total_counts"]), bins=100)
        row[1].axvline(thresholds["log10_umi"], color="r")
        row[1].set_xlabel("log10 UMI per cell")
        row[2].hist(obs["n_genes_by_counts"], bins=100)
        row[2].axvline(thresholds["genes"], color="r")
        row[2].set_xlabel("detected genes per cell")
        row[3].hist(obs["pct_counts_mt"], bins=100)
        row[3].set_xlabel("% mitochondrial counts")
        row[4].hist(obs["pct_counts_ribo"], bins=100)
        row[4].set_xlabel("% ribosomal counts")
        row[0].set_title(f"{label}, {len(obs)} cells")
        for ax in row[1:]:
            ax.set_ylabel("cells")
    fig.tight_layout()
    fig.savefig(fig_dir / "qc_distributions.png", dpi=150)
    plt.close(fig)

    conditions = sorted(adata.obs["condition"].unique(), key=lambda c: (c.split("_")[0], int(c.split("_d")[1])))
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, col, ylabel in zip(
        axes, ["total_counts", "n_genes_by_counts", "pct_counts_mt"], ["UMI", "detected genes", "% mitochondrial"]
    ):
        ax.boxplot([adata.obs.loc[adata.obs["condition"] == c, col] for c in conditions], showfliers=False)
        ax.set_xticks(range(1, len(conditions) + 1), conditions, rotation=90)
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(fig_dir / "qc_per_condition.png", dpi=150)
    plt.close(fig)


def normalise(adata: ad.AnnData) -> ad.AnnData:
    """
    Remove ribosomal genes and MALAT1, keep genes with shared spliced/unspliced counts, and normalise
    spliced to the median spliced count per cell (scvelo, without log); X is the normalised spliced.
    Cell sizes are taken before gene removal (obs "initial_size*", set when velocyto and CellRanger
    outputs were merged). layers["raw_spliced"] keeps the spliced counts.
    """
    removed = adata.var["ribo"] | adata.var_names.isin(OTHER_REMOVED_GENES)
    print(f"Remove {adata.var['ribo'].sum()} ribosomal genes and {OTHER_REMOVED_GENES}")
    adata = adata[:, ~removed].copy()
    adata.layers["raw_spliced"] = adata.layers["spliced"].copy()
    adata.X = adata.layers["spliced"].astype("float32")
    scv.pp.filter_and_normalize(adata, min_shared_counts=MIN_SHARED_COUNTS, layers_normalize="spliced")
    adata.X = adata.layers["spliced"].copy()
    return adata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=REPO_DIR / "data/scrnaseq/1-all_Velocyto.h5ad")
    parser.add_argument("--out", type=Path, default=REPO_DIR / "results/scrnaseq1")
    args = parser.parse_args()

    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.input)
    print(f"{adata.n_obs} cells x {adata.n_vars} genes")
    cells = adata.obs.groupby(["batch", "condition"], observed=True).size().rename("cells")
    cells.to_csv(args.out / "cells_per_condition.csv")

    quality_metrics(adata)
    plot_quality(adata, fig_dir)

    adata = normalise(adata)
    print(f"After gene filtering: {adata.n_obs} cells x {adata.n_vars} genes")
    adata.write_h5ad(args.out / "scrnaseq1_preprocessed.h5ad", compression="gzip")


if __name__ == "__main__":
    main()
