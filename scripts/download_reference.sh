#!/bin/sh
# Download the Ensembl human cDNA FASTA (release 100, the one used to build the kallisto index:
# 190,522 transcripts). scripts/half_life.py reads gene symbols from it. Skipped if already present.
#
# Usage: sh scripts/download_reference.sh
set -eu

RELEASE=100
URL="https://ftp.ensembl.org/pub/release-${RELEASE}/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz"
OUT="$(dirname "$0")/../data/reference/Homo_sapiens.GRCh38.cdna.all.fa.gz"

if [ -f "$OUT" ]; then
    echo "Already present: $OUT"
    exit 0
fi
mkdir -p "$(dirname "$OUT")"
curl -fL --retry 5 -C - -o "$OUT.part" "$URL"
gzip -t "$OUT.part"
mv "$OUT.part" "$OUT"
echo "Downloaded Ensembl release $RELEASE cDNA to $OUT"
