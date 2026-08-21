# Harmonization Pipeline Documentation

This document describes the harmonization phase: turning three raw survey files
(`.sav`) into one anonymized, analysis-ready dataset for machine learning. The phase
is complete and validated. All nine stages run end to end, producing three final
artifacts: one anonymized Parquet (721,385 rows), one encoding schema (JSON), and one
feature availability matrix (CSV).

## 1. Sources and roles

| Source | Role | Notes |
| --- | --- | --- |
| SSGI 2024 | Primary | Main training source. Has pregnancy-complication (KEK, anemia) and stunting-perception modules absent elsewhere. Province only, no district. |
| SSGI 2022 | Cross-period | Cross-time validation. ANC/TTD/maternal-anthropometry modules are baduta-only (age skip pattern). |
| SKI 2023 | Cross-source | Independent external validator. No biomedical module (Hb/retinol/malaria) and no sanitation/toilet module; maternal MUAC empty. |
| ePPGBM Riau | Regional (not processed here) | Province-wide posyandu census (about 9,000 rows). Overlaps the universal set on only 3-4 features. Used downstream for prevalence validation and reduced-feature transfer, not full-feature validation. |

## 2. Final dataset

| Property | Value |
| --- | --- |
| Total rows (3 sources) | 721,385 |
| Rows per source | SSGI22: 334,878 / SSGI24: 300,143 / SKI23: 86,364 |
| Rows per cohort | balita_tua: 460,608 / baduta: 260,777 |
| Stunting prevalence, baduta (unweighted) | 14.27% |
| Stunting prevalence, balita_tua (unweighted) | 22.15% |
| Universal feature set (baduta, 3 sources) | 45 (derived downstream from the availability matrix) |
| Candidate features in the availability matrix | 119 |

Prevalence figures are unweighted raw means. Survey weights, PSU, and strata are
carried as metadata in the Parquet; weighted estimation is a separate analysis step,
not part of harmonization.

## 3. The nine stages

Each source is processed independently through stages 1 to 7, then stacked (stage 8)
and validated (stage 9). Independent processing preserves each survey's structure and
protects SKI 2023 as an independent external validator.

1. **Anonymization.** Drop PII columns. Raw data is processed locally (UU PDP No.
   27/2022).
2. **Variable mapping.** Rename source codes to standard names using dictionary K.
   Unmapped raw columns are filtered out after stage 5.
3. **Anthropometry cleaning.** Remove implausible values (height <40 or >130 cm;
   weight <1 or >40 kg) before z-score computation.
4. **Cohort assignment.** Add a cohort column: baduta (<24 months) vs balita_tua.
   Baduta-module features are set to NaN for balita_tua (structural missingness).
5. **WHO 2006 z-scores.** Compute HAZ/WAZ/WHZ (LMS method) and `stunting_binary`
   (HAZ < -2), with length/height position correction and WHO biological-implausibility
   flagging.
6. **Composite features.** Wealth index (Filmer-Pritchett PCA), ANC totals, dietary
   diversity, TTD compliance, complete-immunization indicator.
7. **Per-source categorical harmonization.** Deterministic recoding of categorical
   values to a canonical scheme (see section 4).
8. **Encoding.** Special missing codes (88, 9999, and so on) to NaN; binary to 0/1;
   ordinal as ordered integers; nominal kept as codes (one-hot deferred to modeling);
   drop columns removed.
9. **MICE imputation, merge, validation.** miceforest (LightGBM) per source and cohort,
   target excluded, structural and unfit columns skipped, missing indicators created;
   then concatenation with a source flag and final validation.

## 4. Categorical value harmonization

Variable-name mapping alone is insufficient: the same concept often carries different
code values across survey waves. Stage 7 aligns these to one canonical scheme, generally
following SSGI 2022. Recoding is deterministic (code-to-code, foreign codes to NaN), so
it introduces no leakage. Key decisions:

- `occupation_mother`: moved from nominal to a binary "mother works" indicator.
- `edu_mother`: collapsed to a canonical five-level ordinal (1 = did not finish primary
  or no schooling, ... , 5 = tertiary) across all three sources.
- `jkn_owned`: reduced to the five SSGI 2022 categories plus a sixth "other insurance"
  category so no row is dropped. This is the most lossy recode and is documented
  explicitly.
- `water_source`, `cooking_fuel`, `toilet_type`, `delivery_place`: SSGI 2024 and SKI
  2023 codes mapped to canonical SSGI 2022 codes.
- Binary items (`ttd_received`, `jkn_used`, `kia_book`, and others) follow SSGI 2022
  (1 = yes, 2 = no), with "do not know" and "not applicable" mapped to NaN.

Value-label expectations are verified in `verifikasi_kamus_sav_133.py`
(EKSPEKTASI_VALUE_LABEL).

## 5. Survey design metadata

Survey weights, PSU, strata, and the household identifier are mapped, filled, and
carried in the Parquet as metadata (role META). All are excluded from MICE and from
special-missing replacement, and none are used as predictors.

| Metadata | Standard name | Per source (SSGI22 / SSGI24 / SKI23) |
| --- | --- | --- |
| Individual/child weight | svy_weight | weight_balita / w_ind_final / w_final |
| Primary sampling unit | svy_psu | PSU / PSU / PSU |
| Stratum | svy_strata | STRATA / STRATA_MASKING / STRATA |
| Household id (cluster) | id_ruta | Cases_ruta / IDRUTA / IDRT |

`id_ruta` is an internal Ministry-of-Health household identifier, not PII, so it is kept
as metadata. Its values are unique per source with differing dtypes, so cluster-aware
cross-validation downstream must group on the pair (source_flag, id_ruta), not id_ruta
alone.

## 6. Availability and the universal feature set

The availability matrix (119 candidate features x 3 sources x 2 cohorts) shows that
coverage is not uniform: the same variable may be available in one source, baduta-only
in another, or absent altogether.

| Group | Count | Meaning |
| --- | --- | --- |
| Universal (3 sources, baduta) | 45 | Strongest foundation and the only set valid for cross-source validation. 39 are also universal for balita_tua; 6 are baduta-only across all three sources (anc_freq_mid_t1/t2/t3, anc_place, ttd_received, ttd_count). |
| Heterogeneous (1-2 sources) | about 55 | Cross-source consistency only testable where a feature is in at least 2 sources. KEK/anemia/perception are SSGI 2024 only. |
| Baduta-only feeding module | 19 | The 1,000-day feeding module (breastfeeding, complementary feeding, dietary items). Baduta models only. |

This matrix (`output_harmonisasi/matriks_ketersediaan.csv`) is the shared source from which each
downstream study derives its own feature set (for example, the 45 universal features for
cross-source validation, or the at-least-two-source intersection for consistency
analysis).

## 7. Anti-leakage protocol for external validation (P3)

Because the main downstream study (P3) claims external validation, leakage is the most
critical risk. Five layers are designed before any modeling code is written.

1. **Target definition (most severe).** `stunting_binary` is derived from
   HAZ = f(height_child_cm, age_child_months, sex_child). Drop `height_child_cm`
   (circular leakage) and `weight_child_kg` (defines WAZ/WHZ and is a concurrent
   condition) as predictors. `age_child_months` and `sex_child` may stay. This removal
   is done at feature selection.
2. **Preprocessing.** MICE excludes the target and is run per source separately, so
   train-on-SSGI / validate-on-SKI is free of imputation leakage. Categorical recoding
   is deterministic.
3. **Class balancing.** SMOTE/ADASYN only inside training folds, never on test folds or
   on SKI 2023 (wrap in an imblearn pipeline inside cross-validation). SKI 2023 is
   evaluated at its natural class distribution.
4. **Cluster leakage.** GroupKFold with group (source_flag, id_ruta) so one household
   never spans train and test folds. Conservative alternative: group on svy_psu.
5. **Tuning leakage.** Bayesian Optimization is tuned on grouped cross-validation inside
   SSGI only; the primary metric is reported once on the untouched SKI 2023.

## 8. Reproduction

```bash
pip install -r requirements.txt
# place SSGI_2022.sav, SSGI_2024.sav, SKI_2023.sav in raw_data/ (or set PATHS)
python harmonisasi_pipeline.py            # writes output_harmonisasi/stunting_harmonized.parquet + skema_encoding.json
python matriks_ketersediaan.py            # optional: availability matrix
python periksa_konvergensi_mice.py        # optional: MICE convergence check
```

Without microdata, generate synthetic test data instead:

```bash
python data_uji/buat_data_uji.py          # writes output_harmonisasi/stunting_harmonized_sintetis.parquet
```

## 9. MICE parameters (final)

`MICE_NUM_DATASETS=1`, `MICE_ITERATIONS=5` (empirically verified sufficient via the
convergence checker), `MICE_MAX_BIN=127`, `force_col_wise=True`, with per-cell memory
release. The ANC block is deliberately not imputed: its missingness is structural and
correlated, so it is carried as a signal through missing indicators rather than filled.
