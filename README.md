# Stunting Harmonization Pipeline

A reproducible, nine-stage harmonization pipeline that transforms three Indonesian
national nutrition survey microdata sources (SSGI 2022, SSGI 2024, SKI 2023) into a
single anonymized, analysis-ready master dataset. This dataset is the shared
foundation for three downstream studies in a doctoral dissertation on
**Applied Explainable AI for Health Risk Prediction**, with childhood stunting in
Riau Province, Indonesia, as the validation domain.

> **DOI:** `10.5281/zenodo.22015038`
## Data availability

The microdata are governed by the Ministry of Health of the Republic of Indonesia
(Kementerian Kesehatan RI) and are **not redistributed** here. See
`raw_data/README.md` for how to request access. The harmonized master Parquet is a
derivative of that microdata and is never committed. To let others run and verify the
downstream pipeline without restricted data, this repository ships a **synthetic test
data generator** (`data_uji/buat_data_uji.py`).

## Repository structure

```
stunting-harmonization/
  README.md
  LICENSE                          # MIT
  requirements.txt
  .gitignore
  .zenodo.json                     # Zenodo record metadata (DOI)
  CITATION.cff                     # citation metadata (GitHub "Cite this repository")
  harmonisasi_pipeline.py          # nine-stage harmonization pipeline (main entry)
  verifikasi_kamus_sav_133.py      # canonical dictionary K + SPSS value-label checks
  zscore_who.py                    # WHO 2006 LMS z-score module (imported by pipeline)
  matriks_ketersediaan.py          # builds the feature availability matrix
  periksa_konvergensi_mice.py      # MICE convergence check
  data_uji/
    buat_data_uji.py               # synthetic Parquet generator (reads the schema)
  docs/
    dokumentasi_harmonisasi.md      # pipeline documentation
  raw_data/                     # (git-ignored) place raw SPSS/Stata microdata here
    README.md                      # how to obtain the microdata
  output_harmonisasi/              # pipeline outputs land here
    skema_encoding.json            # encoding schema (committed; written by the pipeline)
    matriks_ketersediaan.csv       # aggregate availability metadata (committed)
    .gitkeep                       # the master Parquet is written here too but git-ignored
```

## Requirements and installation

```bash
pip install -r requirements.txt
```

The MICE-related pins (`miceforest`, `lightgbm`) must match the versions that produced
the master dataset; `pygrowup` supplies the official WHO 2006 LMS tables used by the
z-score module.

## Usage

### With real microdata

1. Place the three `.sav` files in `raw_data/` and set the paths in `PATHS` inside
   `harmonisasi_pipeline.py` (or run from a folder that contains them).
2. Run the pipeline:
   ```bash
   python harmonisasi_pipeline.py
   ```
   Outputs are written to `output_harmonisasi/`: `stunting_harmonized.parquet` and
   `skema_encoding.json`.
3. Optional diagnostics:
   ```bash
   python matriks_ketersediaan.py        # writes output_harmonisasi/matriks_ketersediaan.csv
   python periksa_konvergensi_mice.py    # verifies MICE convergence at 5 iterations
   ```

### Without microdata (synthetic test data)

```bash
python data_uji/buat_data_uji.py
```

This reads `output_harmonisasi/skema_encoding.json` (and `output_harmonisasi/matriks_ketersediaan.csv`) and writes
`output_harmonisasi/stunting_harmonized_sintetis.parquet`, a small file with the same
schema and structural-missingness pattern as the real output, containing **no real
data values**. Copy it into the input folder to run the full other pipeline
without restricted data. Any statistical relationship in the synthetic data is
artificial and must not be interpreted as a finding.

## Pipeline overview

Each source is processed independently through stages 1 to 7, then stacked (stage 8) and
validated (stage 9). Independent processing preserves each survey's structure and keeps
SKI 2023 usable as an independent external validator downstream.

1. Anonymization (drop PII; local-only processing, UU PDP No. 27/2022).
2. Variable mapping to standard names via dictionary K.
3. Anthropometry cleaning (implausible height/weight removed).
4. Cohort assignment: baduta (<24 months) vs balita_tua (24-59 months).
5. WHO 2006 LMS z-scores (HAZ/WAZ/WHZ) and `stunting_binary` (HAZ < -2).
6. Composite features (wealth index, ANC totals, dietary diversity, and so on).
7. Per-source categorical value harmonization (canonical recoding).
8. Encoding (binary to 0/1, ordinal integers, nominal kept as codes for later one-hot).
9. MICE imputation (per source and cohort, target excluded; missing indicators added),
   then concatenation and validation.

See `docs/dokumentasi_harmonisasi.md` for the full description and the methodological
decisions to cite in a methods section.

## Feature availability and downstream feature sets

The availability matrix (`output_harmonisasi/matriks_ketersediaan.csv`) records, for each
candidate feature, in which of the six cells (three sources x two cohorts) it is
actually available. This matrix is the shared artifact from which each downstream study
derives its own feature set, according to its own rule. For example, cross-source
validation (P3) uses the 45 features available in all three sources for the baduta
cohort, while consistency analysis (P2) uses features present in at least two sources.

Note on leakage: the 45 universal set is defined by **availability**, so it includes
`height_child_cm` and `weight_child_kg`. Because those two anthropometric measurements
define the target (HAZ/WAZ/WHZ), any external-validation model must **drop them as
predictors** even though they are available (leaving 43). This exclusion is applied at
modeling time in the downstream repository, not here.

## License and citation

Released under the MIT License (see `LICENSE`). If you use this code, please cite the
DOI-bearing release (see `CITATION.cff` and `.zenodo.json`).
