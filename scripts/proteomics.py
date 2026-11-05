"""TMT proteomics of WT and DMD mutant cells, day 5 to 14 (Fig S26).

Protein intensities (EDyP export, 3 replicates per genotype and day) are mapped to Ensembl genes,
converted to copies per cell with the histone proteomic ruler (Wisniewski et al. 2014), then averaged
over replicates.

Usage:
    python scripts/proteomics.py --input data/proteomics/Proteomic_result.csv --out results/proteomics
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent

GENOTYPES = ["WT", "Mut"]
DAYS = [5, 6, 7, 8, 9, 10, 12, 14]
REPLICATES = ["R1", "R2", "R3"]
AVOGADRO = 6.02214086e23
MIN_REPLICATES = 2  # replicates with a value needed to compute a mean

# Ensembl ids searched by hand for proteins without (or with several) HGNC Ensembl ids,
# keyed by protein group id of the EDyP export
MANUAL_ENSEMBL_IDS = {
    422: "ENSG00000204525",  # HLA-C
    423: "ENSG00000206503",  # HLA-A
    193: "ENSG00000160221",
    722: "ENSG00000075790",
    806: "ENSG00000100836",
    1061: "ENSG00000115966",
    2044: "ENSG00000205544",
    2675: "ENSG00000160201",
    2676: "ENSG00000160200",
    2881: "ENSG00000288722",
    3393: "ENSG00000278637",
    4182: "ENSG00000167552",
    5515: "ENSG00000232119",
    5630: "ENSG00000146963",
}
DROPPED_PROTEINS = [412]  # several Ensembl ids, none could be chosen

# Genes shown in Fig S26 (PDGFRA and PTN are not detected)
GENES_OF_INTEREST = {
    "ENO1": "ENSG00000074800",
    "ID1": "ENSG00000125968",
    "MIF": "ENSG00000240972",
    "PRTG": "ENSG00000166450",
}


def intensity_columns() -> list[str]:
    return [f"raw{rep}-{geno}{day}" for rep in REPLICATES for geno in GENOTYPES for day in DAYS]


def load_intensities(csv: Path, mapping_csv: Path) -> pd.DataFrame:
    """Read raw intensities, gene names, molecular weight and Ensembl gene id per protein group."""
    columns = intensity_columns()
    proteins = pd.read_csv(csv, dtype=str, index_col="id")
    proteins.index = proteins.index.astype(int)
    proteins = proteins.sort_index()  # for protein groups summed into one gene, metadata come from the lowest id
    proteins = proteins[["Gene names", "Mol. weight [kDa]", *columns]]
    # Decimal commas in the export
    for col in ["Mol. weight [kDa]", *columns]:
        proteins[col] = proteins[col].str.replace(",", ".").astype(float)
    # Day 14 WT and Mut samples were inverted in the export
    swap = {f"raw{rep}-{a}14": f"raw{rep}-{b}14" for rep in REPLICATES for a, b in [("WT", "Mut"), ("Mut", "WT")]}
    proteins = proteins.rename(columns=swap)[["Gene names", "Mol. weight [kDa]", *columns]]

    mapping = pd.read_csv(mapping_csv, index_col="id")
    proteins["ensembl_id"] = mapping.loc[proteins.index, "main_ensembl_id"]
    return proteins


def assign_genes(proteins: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one Ensembl gene per protein group, then sum protein groups sharing a gene.
    Protein groups with several Ensembl ids and several gene names are ambiguous and removed.
    """
    manual = pd.Series(MANUAL_ENSEMBL_IDS)
    unknown = proteins.index.isin(manual.index) & proteins["ensembl_id"].isna()
    proteins.loc[unknown, "ensembl_id"] = manual[proteins.index[unknown]].values
    proteins = proteins[proteins["ensembl_id"].notna()]

    multi_ensembl = proteins["ensembl_id"].str.contains(";")
    multi_name = proteins["Gene names"].str.contains(";", na=False)
    proteins = proteins[~(multi_ensembl & multi_name)].drop(DROPPED_PROTEINS)
    fixed = proteins.index.isin(manual.index)
    proteins.loc[fixed, "ensembl_id"] = manual[proteins.index[fixed]].values
    assert not proteins["ensembl_id"].str.contains(";").any()

    columns = intensity_columns()
    genes = proteins.groupby("ensembl_id").agg(
        symbol=("Gene names", "first"), mol_weight_kda=("Mol. weight [kDa]", "first")
    )
    # Sum intensities of protein groups of the same gene (all missing stays missing)
    genes[columns] = proteins.groupby("ensembl_id")[columns].sum(min_count=1)
    genes["symbol"] = genes["symbol"].str.split(";").str[0]
    return genes


def histone_ruler(genes: pd.DataFrame, histones_csv: Path) -> pd.DataFrame:
    """
    Copies per cell: N_A * I_protein / (MW_protein * sum of histone intensities).
    Histone mass per cell is set to 1, so values are proportional to copies per cell.
    Only histones quantified in every sample are used.
    """
    columns = intensity_columns()
    histone_ids = pd.read_csv(histones_csv)["Ensembl Gene ID"]
    histones = genes.loc[genes.index.isin(histone_ids), columns].dropna()
    total_histone = histones.sum(axis=0)
    print(f"Histone ruler: {len(histones)} histone genes quantified in all samples")
    copies = AVOGADRO * genes[columns].div(genes["mol_weight_kda"], axis=0) / total_histone
    return copies


def to_long(copies: pd.DataFrame, symbols: pd.Series) -> pd.DataFrame:
    long = copies.rename_axis("gene_id").reset_index().melt(id_vars="gene_id", var_name="sample", value_name="copies")
    parts = long["sample"].str.extract(r"raw(?P<replicate>R\d)-(?P<genotype>WT|Mut)(?P<day>\d+)")
    long = pd.concat([long[["gene_id"]], parts, long[["copies"]]], axis=1)
    long["day"] = long["day"].astype(int)
    long.insert(1, "symbol", symbols.loc[long["gene_id"]].values)
    return long


def replicate_mean(long: pd.DataFrame) -> pd.DataFrame:
    """Mean over replicates per gene, genotype and day; NaN if fewer than MIN_REPLICATES values."""
    grouped = long.groupby(["gene_id", "symbol", "genotype", "day"], sort=False, dropna=False)["copies"]
    mean = grouped.agg(mean_copies="mean", n_replicates="count").reset_index()
    mean.loc[mean["n_replicates"] < MIN_REPLICATES, "mean_copies"] = np.nan
    return mean


def plot_genes_of_interest(long: pd.DataFrame, mean: pd.DataFrame, fig_dir: Path):
    """Protein copies per cell over days for WT and mutant (Fig S26, protein part)."""
    fig, axes = plt.subplots(1, len(GENES_OF_INTEREST), figsize=(4 * len(GENES_OF_INTEREST), 3.5))
    colors = {"WT": "tab:blue", "Mut": "tab:red"}
    for ax, (symbol, gene_id) in zip(axes, GENES_OF_INTEREST.items()):
        for genotype in GENOTYPES:
            points = long[(long["gene_id"] == gene_id) & (long["genotype"] == genotype)]
            line = mean[(mean["gene_id"] == gene_id) & (mean["genotype"] == genotype)].sort_values("day")
            ax.scatter(points["day"], points["copies"], color=colors[genotype], alpha=0.4, s=12)
            ax.plot(line["day"], line["mean_copies"], color=colors[genotype], marker="o", label=genotype)
        ax.set_title(symbol)
        ax.set_xlabel("day of differentiation")
        ax.set_xticks(DAYS)
    axes[0].set_ylabel("protein copies per cell (a.u.)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "protein_genes_of_interest.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=REPO_DIR / "data/proteomics/Proteomic_result.csv")
    parser.add_argument("--mapping", type=Path, default=REPO_DIR / "resources/proteomics_protein_to_ensembl.csv")
    parser.add_argument("--histones", type=Path, default=REPO_DIR / "resources/histones_MS_HistoneDB.csv")
    parser.add_argument("--out", type=Path, default=REPO_DIR / "results/proteomics")
    args = parser.parse_args()

    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    proteins = load_intensities(args.input, args.mapping)
    genes = assign_genes(proteins)
    print(f"{len(proteins)} protein groups -> {len(genes)} genes")

    copies = histone_ruler(genes, args.histones)
    long = to_long(copies, genes["symbol"])
    long.to_csv(args.out / "protein_copies_per_cell.csv", index=False)
    mean = replicate_mean(long)
    mean.to_csv(args.out / "protein_mean.csv", index=False)

    plot_genes_of_interest(long, mean, fig_dir)


if __name__ == "__main__":
    main()
