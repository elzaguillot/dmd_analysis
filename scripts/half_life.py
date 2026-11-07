"""Bulk RNA half-life (actinomycin D) analysis.

UC3-4 cells treated with actinomycin D at day 5, 7 and 9 of differentiation and collected
0, 1, 2, 3 and 4 hours after treatment. Transcript abundances  are aggregated per
gene, and an exponential decay y = a * exp(-d0 * t) is fitted for each gene and day to
estimate the mRNA degradation rate d0 (h^-1) and half-life ln(2) / d0.

Usage:
    python scripts/half_life.py --kallisto-dir data/half_life/kallisto \
        --fasta data/reference/Homo_sapiens.GRCh38.cdna.all.fa.gz --out results/half_life

The FASTA is downloaded with `sh scripts/download_reference.sh`.
"""

import argparse
import gzip
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit

REPO_DIR = Path(__file__).resolve().parent.parent

DAYS = ["D5", "D7", "D9"]
TIME_POINTS_H = [0, 1, 2, 3, 4]
MIN_D0 = 0.01  # d0 below this value is not estimable from a 4 h kinetic
MIN_R2 = 0.6  # goodness of fit threshold used to keep a d0 estimate
# Genes discussed in the paper, pinned by Ensembl id (MIF also has a copy on an alternate haplotype scaffold)
GENES_OF_INTEREST = {
    "ENO1": "ENSG00000074800",
    "MIF": "ENSG00000240972",
    "PRTG": "ENSG00000166450",
    "ID1": "ENSG00000125968",
    "PDGFRA": "ENSG00000134853",
    "PTN": "ENSG00000105894",
}


def read_gene_symbols(fasta: Path) -> pd.Series:
    """Map Ensembl gene id (without version) to gene symbol from Ensembl cDNA FASTA headers."""
    opener = gzip.open if fasta.suffix == ".gz" else open
    symbols = {}
    with opener(fasta, "rt") as fa:
        for line in fa:
            if not line.startswith(">"):
                continue
            fields = dict(f.split(":", 1) for f in line[1:].split() if ":" in f)
            if "gene" in fields and "gene_symbol" in fields:
                symbols[fields["gene"].split(".")[0]] = fields["gene_symbol"]
    return pd.Series(symbols, name="symbol")


def tpm_per_gene(kallisto_dir: Path, tr2g: pd.DataFrame) -> pd.DataFrame:
    """Sum kallisto transcript TPM per gene (same as tximport(type="kallisto")$abundance)."""
    tx_to_gene = tr2g.set_index("transcript_id")["gene_id"]
    samples = [f"{day}_T{t}" for day in DAYS for t in TIME_POINTS_H]
    tpm = {}
    for sample in samples:
        abundance = pd.read_csv(kallisto_dir / sample / "abundance.tsv", sep="\t", index_col="target_id")
        abundance = abundance[abundance.index.isin(tx_to_gene.index)]
        tpm[sample] = abundance["tpm"].groupby(tx_to_gene.loc[abundance.index].values).sum()
    return pd.DataFrame(tpm)


def exp_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Exponential function y = a * exp(-b * x)"""
    return a * np.exp(-b * x)


def compute_r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    """R squared of a curve fitting."""
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot


def fit_d0(expression: np.ndarray, time_points: np.ndarray) -> tuple[float, float, float]:
    """
    Fit an exponential decay to the expression of one gene on one day.
    Args:
        expression: expression values (TPM) at each time point
        time_points: time after actinomycin D treatment (hours)

    Returns:
        d0 (NaN if below MIN_D0), amplitude a (TPM) and r squared; all NaN if the fit is not possible
    """
    # Need a non-zero signal with at least three non zero values
    if expression.max() == 0 or np.count_nonzero(expression) < 3:
        return np.nan, np.nan, np.nan
    # Normalise by the maximum to avoid overflow in exp
    y = expression / expression.max()
    try:
        popt, _ = curve_fit(exp_func, time_points, y)
    except RuntimeError:
        return np.nan, np.nan, np.nan
    d0 = popt[1] if popt[1] >= MIN_D0 else np.nan
    r_squared = compute_r_squared(y, exp_func(time_points, *popt))
    return d0, popt[0] * expression.max(), r_squared


def compute_d0_table(tpm: pd.DataFrame, symbols: pd.Series) -> pd.DataFrame:
    """Fit d0 for every gene and day, return one row per gene."""
    time_points = np.array(TIME_POINTS_H, dtype=float)
    table = pd.DataFrame(index=tpm.index)
    table.index.name = "gene_id"
    table["symbol"] = symbols.reindex(tpm.index).fillna(pd.Series(tpm.index, index=tpm.index))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        for day in DAYS:
            values = tpm[[f"{day}_T{t}" for t in TIME_POINTS_H]].to_numpy()
            fits = np.array([fit_d0(row, time_points) for row in values])
            table[f"d0_{day}"] = fits[:, 0]
            table[f"a_{day}"] = fits[:, 1]
            table[f"r2_{day}"] = fits[:, 2]
            table[f"half_life_h_{day}"] = np.log(2) / table[f"d0_{day}"]
            table[f"pass_r2_{day}"] = table[f"r2_{day}"] > MIN_R2
    return table


def plot_distributions(d0_table: pd.DataFrame, fig_dir: Path):
    """Histograms of d0, half-life and r squared per day (d0 and half-life for fits passing the r2 filter)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for day in DAYS:
        passed = d0_table[d0_table[f"pass_r2_{day}"]]
        axes[0].hist(passed[f"d0_{day}"].dropna(), bins=100, range=(0, 2), histtype="step", label=day)
        axes[1].hist(passed[f"half_life_h_{day}"].dropna(), bins=100, range=(0, 30), histtype="step", label=day)
        axes[2].hist(d0_table[f"r2_{day}"].dropna(), bins=100, range=(0, 1), histtype="step", label=day)
    axes[0].set_xlabel("d0 (h$^{-1}$), r$^2$ > %.1f" % MIN_R2)
    axes[1].set_xlabel("half-life (h), r$^2$ > %.1f" % MIN_R2)
    axes[2].set_xlabel("r$^2$")
    for ax in axes:
        ax.set_ylabel("genes")
        ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "d0_half_life_r2_distributions.png", dpi=150)
    plt.close(fig)


def plot_genes_of_interest(tpm: pd.DataFrame, d0_table: pd.DataFrame, fig_dir: Path):
    """Decay fits for the genes of interest, and their expression before treatment (T0) across days."""
    genes = pd.Series(GENES_OF_INTEREST)
    time_points = np.array(TIME_POINTS_H, dtype=float)
    smooth_t = np.linspace(0, max(TIME_POINTS_H), 50)

    fig, axes = plt.subplots(len(genes), len(DAYS), figsize=(10, 2.2 * len(genes)), squeeze=False)
    for i, (symbol, gene_id) in enumerate(genes.items()):
        for j, day in enumerate(DAYS):
            ax = axes[i, j]
            ax.scatter(time_points, tpm.loc[gene_id, [f"{day}_T{t}" for t in TIME_POINTS_H]])
            fit = d0_table.loc[gene_id]
            if not np.isnan(fit[f"d0_{day}"]):
                ax.plot(smooth_t, exp_func(smooth_t, fit[f"a_{day}"], fit[f"d0_{day}"]), "r")
            ax.set_title(f"{symbol} {day}: d0={fit[f'd0_{day}']:.3f}, r$^2$={fit[f'r2_{day}']:.2f}", fontsize=8)
            if j == 0:
                ax.set_ylabel("TPM")
            if i == len(genes) - 1:
                ax.set_xlabel("time after actinomycin D (h)")
    fig.tight_layout()
    fig.savefig(fig_dir / "decay_fits_genes_of_interest.png", dpi=150)
    plt.close(fig)

    t0 = tpm.loc[genes.values, [f"{day}_T0" for day in DAYS]]
    t0.index = genes.index
    t0.columns = DAYS
    fig, ax = plt.subplots(figsize=(5, 4))
    for symbol, row in t0.iterrows():
        ax.plot([int(d[1:]) for d in DAYS], row.values, marker="o", label=symbol)
    ax.set_xlabel("day of differentiation")
    ax.set_ylabel("TPM before actinomycin D (T0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "t0_expression_genes_of_interest.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kallisto-dir", type=Path, default=REPO_DIR / "data/half_life/kallisto")
    parser.add_argument("--tr2g", type=Path, default=REPO_DIR / "resources/tr2g_kallisto_index.csv.gz")
    parser.add_argument("--fasta", type=Path, default=REPO_DIR / "data/reference/Homo_sapiens.GRCh38.cdna.all.fa.gz")
    parser.add_argument("--out", type=Path, default=REPO_DIR / "results/half_life")
    args = parser.parse_args()

    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    tr2g = pd.read_csv(args.tr2g)
    tpm = tpm_per_gene(args.kallisto_dir, tr2g)
    tpm.index.name = "gene_id"
    tpm.to_csv(args.out / "tpm_per_gene.csv")
    print(f"TPM table: {tpm.shape[0]} genes x {tpm.shape[1]} samples")

    if not args.fasta.exists():
        parser.error(f"{args.fasta} not found, download it with: sh scripts/download_reference.sh")
    symbols = read_gene_symbols(args.fasta)
    d0_table = compute_d0_table(tpm, symbols)
    d0_table.to_csv(args.out / "d0_per_gene.csv")
    for day in DAYS:
        n_pass = (d0_table[f"pass_r2_{day}"] & d0_table[f"d0_{day}"].notna()).sum()
        print(f"{day}: d0 estimated with r2 > {MIN_R2} for {n_pass} genes")

    plot_distributions(d0_table, fig_dir)
    plot_genes_of_interest(tpm, d0_table, fig_dir)


if __name__ == "__main__":
    main()
