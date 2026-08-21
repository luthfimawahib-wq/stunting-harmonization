# Raw microdata (not included)

This folder is intentionally empty in version control. It is where the raw survey
microdata files must be placed to run the pipeline on real data.

## Files expected here

- `SSGI_2022.sav` (primary cross-period source)
- `SSGI_2024.sav` (primary training source)
- `SKI_2023.sav` (independent external validator)

The exact file names must match the `PATHS` dictionary in
`../harmonisasi_pipeline.py`, or you can edit `PATHS` to point at your local names.

## How to obtain the microdata

The microdata are governed by the Ministry of Health of the Republic of Indonesia
(Kementerian Kesehatan RI). They are restricted derivatives and are **not
redistributed** in this repository. Access is granted through an official data request
to the responsible institution:

- Survei Status Gizi Indonesia (SSGI) and Survei Kesehatan Indonesia (SKI): submit a
  data request to the Badan Kebijakan Pembangunan Kesehatan (BKPK), Kementerian
  Kesehatan RI, following their data-access procedure.

Raw `.sav` and `.dta` files, along with the derived master Parquet, are covered by the
project `.gitignore` and must remain on your local machine only, in line with UU PDP
No. 27/2022 (Indonesian Personal Data Protection Law).

## Running without the microdata

If you do not have access to the microdata, use the synthetic test-data generator
instead:

```bash
python ../data_uji/buat_data_uji.py
```

It produces a schema-compatible synthetic Parquet in `../output_harmonisasi/` that
lets the downstream pipeline run without any restricted data.
