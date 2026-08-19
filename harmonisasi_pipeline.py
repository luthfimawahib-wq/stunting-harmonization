import os
import re
import sys
import importlib.util
from datetime import datetime

import numpy as np
import pandas as pd

PATHS = {
    "ssgi22": "raw_data/SSGI_2022.sav",
    "ssgi24": "raw_data/SSGI_2024.sav",
    "ski23":  "raw_data/SKI_2023.sav",
}
KAMUS_SCRIPT = "verifikasi_kamus_sav_133.py"
DIR_REF_WHO  = "who_lms"
DIR_OUTPUT   = "output_harmonisasi"
PARQUET_FINAL = os.path.join(DIR_OUTPUT, "stunting_harmonized.parquet")

KODE_MISSING_KHUSUS = {88, 98, 99, 888, 998, 999, 9999, 99999}

RANGE_ANTRO = {
    "height_child_cm": (40.0, 130.0),
    "weight_child_kg": (1.0, 40.0),
    "age_child_months": (0, 60),
}
RANGE_ZSCORE = {"haz_score": (-6, 6), "waz_score": (-6, 5), "whz_score": (-5, 5)}

AMBANG_BADUTA = 24

FITUR_MODUL_BADUTA = {
    "food_water", "food_formula", "food_cereal", "food_vit_a_veg", "food_green_veg",
    "food_vit_a_fruit", "food_organ_meat", "food_meat", "food_egg", "food_fish",
    "food_legume", "meal_frequency", "food_diversity_score",
    "breastfed_current", "mpasi_age_months", "prelacteal_feed",
    "imd_skin_contact", "colostrum_action", "breastfed_ever", "imd_duration",
    "weaning_age_months",
}

SKEMA_ENCODING = {
    "biner": [
        "stunting_binary",
        "area_type", "sex_child",
        "occupation_mother",
        "anc_received", "anc_lila_measured", "anc_hb_tested", "pregnancy_class",
        "kek_flag", "anemia_preg_flag", "hypertension_preg", "diabetes_preg",
        "ttd_received", "pmt_received", "mms_received",
        "imd_skin_contact", "breastfed_ever", "breastfed_current",
        "food_water", "food_formula", "food_cereal", "food_vit_a_veg", "food_green_veg",
        "food_vit_a_fruit", "food_organ_meat", "food_meat", "food_egg", "food_fish",
        "food_legume",
        "imm_hepb0", "imm_bcg", "imm_dpt1", "imm_dpt2", "imm_dpt3", "imm_dpt_boost",
        "imm_pcv", "imm_polio", "imm_measles_9mo", "imm_measles_boost",
        "ispa_1month", "diarrhea_1month", "pneumonia_1year", "tb_1year", "worm_1year",
        "jkn_used",
        "kia_book",
        "asset_gas", "asset_washing_machine", "asset_fridge", "asset_phone",
        "asset_computer", "asset_tv", "asset_motorcycle", "asset_car",
        "asset_gold", "asset_land", "asset_livestock",
        "bansos_kks", "bansos_pkh", "bansos_bpnt", "bansos_blt",
        "malaria_rdt",
        "TTD_compliance", "imunisasi_lengkap",
        "prelacteal_feed",
        "sanitation_own",
        "height_measure_12mo",
        "kpsp_sdidtk",
        "know_stunting",
    ],
    "ordinal": [
        "edu_mother",
        "percep_hereditary", "percep_asi", "percep_ttd",
        "percep_obese_misc", "percep_cognition", "percep_environment",
    ],
    "nominal": [
        "delivery_place", "delivery_assistant", "delivery_method",
        "marital_status", "anc_place", "colostrum_action",
        "water_source", "toilet_type", "feces_disposal",
        "building_ownership", "cooking_fuel", "lighting_source",
        "jkn_owned",
        "know_info_source",
        "immunity_serology",
    ],
    "kontinu": [
        "haz_score", "waz_score", "whz_score",
        "age_child_months", "weight_child_kg", "height_child_cm",
        "lila_child_cm", "weight_gain_trend",
        "birth_weight_g", "birth_length_cm", "gestational_age_wk", "head_circ_birth_cm",
        "age_mother_yr", "gravida",
        "height_mother_cm", "weight_mother_kg", "lila_mother_cm",
        "anc_freq_doc_t1", "anc_freq_doc_t2", "anc_freq_doc_t3",
        "anc_freq_mid_t1", "anc_freq_mid_t2", "anc_freq_mid_t3",
        "ttd_count", "imd_duration",
        "weaning_age_months", "mpasi_age_months", "meal_frequency",
        "imm_vit_a_count", "weigh_freq_12mo", "water_fetch_time", "floor_area_m2",
        "hb_level_gdl", "retinol_level",
        "wealth_index", "ANC_total_visit", "food_diversity_score",
    ],
    "meta": [
        "source_flag", "kohort", "province", "district",
        "svy_weight", "svy_psu", "svy_strata",
        "id_ruta",
    ],
    "drop": [
        "pii_identitas", "subdistrict", "village", "health_facility",
        "child_birth_date", "measure_date",
        "measure_position",
    ],
}

REKODE_BINER_KHUSUS = {
    "kia_book": ([1, 2], [3, 4]),
}

REKODE_POSISI_PER_SUMBER = {
    "ssgi22": {1: 2, 2: 1},
    "ski23":  {1: 2, 2: 1},
}

REKODE_BINER_PER_SUMBER = {
    "ssgi22": {
        "ttd_received": ([1, 2, 3], [4]),
        "jkn_used": ([1, 2, 3], [4]),
    },
    "ssgi24": {
        **{v: ([1, 2, 3], [4]) for v in (
            "imm_hepb0", "imm_bcg", "imm_dpt1", "imm_dpt2", "imm_dpt3",
            "imm_dpt_boost", "imm_measles_9mo", "imm_measles_boost")},
        **{v: ([1, 2], [3]) for v in (
            "ispa_1month", "diarrhea_1month", "pneumonia_1year", "tb_1year")},
        "sanitation_own": ([1, 2, 3], [4]),
        "occupation_mother": ([2, 3, 4, 5, 6, 7, 8, 9], [1]),
    },
    "ski23": {
        "occupation_mother": ([2, 3, 4, 5, 6, 7, 8, 9], [1]),
        "kia_book": ([1, 2, 3, 4, 5], [6, 7]),
    },
}

REKODE_NOMINAL_PER_SUMBER = {
    "ssgi22": {
        "jkn_owned":    {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    },
    "ssgi24": {
        "water_source": {1: 1, 2: 2, 3: 3, 4: 11, 5: 9, 6: 10, 7: 7, 8: 8, 9: 6, 10: 13, 11: 14},
        "cooking_fuel": {1: 8, 2: 7, 3: 6, 4: 5, 5: 3, 6: 2, 7: 1},
        "toilet_type":  {1: 1, 2: 2, 3: 3, 4: 4, 5: 6},
        "jkn_owned":    {1: 1, 2: 2, 3: 1, 4: 1, 5: 1, 6: 2, 7: 1, 8: 3, 9: 4, 10: 4,
                         11: 4, 12: 4, 13: 4, 14: 4, 15: 4, 16: 6, 17: 1, 18: 2,
                         19: 1, 20: 1, 21: 1, 22: 2, 23: 1, 24: 3, 25: 4, 26: 4, 27: 4,
                         28: 4, 29: 4, 30: 4, 31: 4, 32: 5},
    },
    "ski23": {
        "water_source": {1: 1, 2: 2, 3: 3, 4: 11, 5: 9, 6: 10, 7: 7, 8: 8, 9: 6, 10: 13, 11: 4, 12: 5, 13: 12},
        "cooking_fuel": {1: 1, 2: 2, 3: 3, 4: 5, 5: 6, 6: 7, 7: 8},
        "delivery_place": {1: 1, 2: 1, 3: 2, 4: 3, 5: 5, 6: 4, 7: 7, 8: 8, 9: 9},
        "jkn_owned":    {1: 1, 2: 2, 4: 1, 5: 1, 8: 3, 10: 4, 16: 6, 32: 5, 99: 6},
    },
}

REKODE_ORDINAL_PER_SUMBER = {
    "ssgi24": {
        "edu_mother": {1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 5},
    },
    "ski23": {
        "edu_mother": {0: None, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 5},
    },
}

MICE_KECUALIKAN = {
    "stunting_binary", "haz_score", "waz_score", "whz_score",
    "source_flag", "kohort", "province", "district",
    "svy_weight", "svy_psu", "svy_strata",
    "id_ruta",
}
MICE_JANGAN_IMPUTASI = {
    "ANC_total_visit",
    "anc_freq_doc_t1", "anc_freq_doc_t2", "anc_freq_doc_t3",
    "anc_freq_mid_t1", "anc_freq_mid_t2", "anc_freq_mid_t3",
}
MICE_NUM_DATASETS = 1
MICE_ITERATIONS = 5
MICE_RANDOM_STATE = 42
MICE_MIN_DATA_IN_LEAF = 50
MICE_MIN_TERISI = 100
MICE_MAX_BIN = 127
MICE_AMBANG_INDIKATOR = 0.10

def bersihkan_nama_kolom(df):
    return df.rename(columns={c: re.sub(r"[^\x00-\x7F]", "", str(c)).strip()
                              for c in df.columns})

def baca_sav(path):
    import pyreadstat
    df, meta = pyreadstat.read_sav(path, encoding="latin1")
    df = bersihkan_nama_kolom(df)
    return df, meta

def muat_kamus(script_path=KAMUS_SCRIPT):
    import types
    asli = sys.modules.get("pyreadstat")
    stub = types.ModuleType("pyreadstat")
    stub.read_sav = lambda *a, **k: (None, None)
    sys.modules["pyreadstat"] = stub
    try:
        spec = importlib.util.spec_from_file_location("kamus", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        K, SOURCE_COL = mod.K, mod.SOURCE_COL
    finally:
        if asli is not None:
            sys.modules["pyreadstat"] = asli
        else:
            sys.modules.pop("pyreadstat", None)
    return K, SOURCE_COL

_INVIS = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u00a0]")

def _norm_kol(s):
    return _INVIS.sub("", str(s)).strip().lower()

def _bersih_kode(cell):
    s = _INVIS.sub("", str(cell)).strip().replace("*", "")
    s = re.split(r"\s*[(/]", s)[0].strip()
    return s

def _cocok_kolom(kode, nama, kolom_list, kolom_norm):
    nk = _norm_kol(kode)
    if not nk:
        return None
    if nk in kolom_norm:
        return kolom_norm[nk]
    kand = [(c, m.group(1)) for c in kolom_list
            if (m := re.match(rf"^{re.escape(nk)}_(.+)$", _norm_kol(c)))]
    if not kand:
        return None
    if len(kand) == 1:
        return kand[0][0]
    nm = nama.lower()
    subjek = ("ibu" if ("mother" in nm or "ibu" in nm)
              else "ayah" if ("father" in nm or "ayah" in nm)
              else "anak")
    for c, peran in kand:
        if peran == subjek:
            return c
    return None

def bangun_peta_rename(K, SOURCE_COL, source, kolom_tersedia):
    col = SOURCE_COL[source]
    kolom_list = list(kolom_tersedia)
    kolom_norm = {_norm_kol(c): c for c in kolom_list}
    peta, klaim, lewat = {}, {}, []
    for row in K:
        nama, kode_raw = row[0], str(row[col]).strip()
        if kode_raw in ("—", "", "nan") or kode_raw.upper().startswith("DERIVASI"):
            continue
        kode = _bersih_kode(kode_raw)
        aktual = _cocok_kolom(kode, nama, kolom_list, kolom_norm)
        if aktual is None:
            lewat.append((nama, kode_raw))
            continue
        if aktual in klaim and klaim[aktual] != nama:
            lewat.append((nama, f"{kode_raw}->{aktual} bentrok {klaim[aktual]}"))
            continue
        peta[aktual] = nama
        klaim[aktual] = nama
    return peta, lewat

def kelompok_kolom(K, prefix=None, peran=None, kategori=None):
    out = []
    for row in K:
        nama, kat, _, _, _, _, _, prn = row
        if prefix and not nama.startswith(prefix):
            continue
        if peran and prn != peran:
            continue
        if kategori and kat != kategori:
            continue
        out.append(nama)
    return out

def tahap1_anonimisasi(df, source, K, SOURCE_COL):
    col = SOURCE_COL[source]
    kode_pii = [str(row[col]).strip() for row in K if row[7] == "HAPUS_PRIVASI"]
    kode_pii = [k for k in kode_pii if k not in ("—", "", "nan")]
    drop = [c for c in df.columns if c in kode_pii]
    df = df.drop(columns=drop, errors="ignore")
    print(f"    [1] Anonimisasi: {len(drop)} kolom PII dibuang")
    return df

def tahap2_pemetaan(df, source, K, SOURCE_COL):
    peta, lewat = bangun_peta_rename(K, SOURCE_COL, source, df.columns)
    df = df.rename(columns=peta)
    msg = f"    [2] Pemetaan: {len(peta)} kolom -> nama standar"
    if lewat:
        msg += f" | {len(lewat)} kode tak tercocokkan"
    print(msg)
    if lewat:
        contoh = ", ".join(f"{n}({k})" for n, k in lewat[:8])
        print(f"        tak tercocokkan (cek bila penting): {contoh}")
    return df

def saring_kolom_standar(df, K):
    nama_standar = [row[0] for row in K]
    turunan = ["wealth_index", "ANC_total_visit", "food_diversity_score",
               "TTD_compliance", "imunisasi_lengkap",
               "haz_score", "waz_score", "whz_score", "stunting_binary",
               "kohort", "source_flag"]
    simpan = [c for c in df.columns if c in set(nama_standar + turunan)]
    df = df[simpan].copy()
    print(f"        Penyaringan: {df.shape[1]} kolom standar dipertahankan")
    return df

def tambah_kohort(df):
    if "age_child_months" not in df.columns:
        df["kohort"] = pd.NA
        print("        Kohort: age_child_months tidak ada -> kohort = NA")
        return df
    umur = pd.to_numeric(df["age_child_months"], errors="coerce")
    df["kohort"] = np.where(umur < AMBANG_BADUTA, "baduta", "balita_tua")
    df.loc[umur.isna(), "kohort"] = pd.NA
    mask_tua = df["kohort"] == "balita_tua"
    for c in (FITUR_MODUL_BADUTA & set(df.columns)):
        df.loc[mask_tua, c] = np.nan
    n_bad = int((df["kohort"] == "baduta").sum())
    print(f"        Kohort: baduta={n_bad:,} | balita_tua={int(mask_tua.sum()):,}")
    return df

def diagnosa_missing_struktural(df, ambang_korelasi=0.5):
    if "kohort" not in df.columns:
        return
    bad = df[df["kohort"] == "baduta"]
    tua = df[df["kohort"] == "balita_tua"]
    baris = []
    for c in df.columns:
        if c in ("kohort", "source_flag"):
            continue
        m_bad, m_tua = bad[c].isna().mean(), tua[c].isna().mean()
        if abs(m_tua - m_bad) >= ambang_korelasi:
            baris.append((c, round(m_bad, 3), round(m_tua, 3)))
    if baris:
        print("\n  Kandidat fitur MODUL-BADUTA (missing baduta << balita_tua):")
        for c, mb, mt in sorted(baris, key=lambda x: x[2] - x[1], reverse=True):
            tanda = "✓ sudah ditandai" if c in FITUR_MODUL_BADUTA else "← PERIKSA"
            print(f"    {c:24} baduta={mb:.0%}  balita_tua={mt:.0%}  {tanda}")

def tahap3_koreksi_antropometri(df):
    n_total = 0
    for kol, (lo, hi) in RANGE_ANTRO.items():
        if kol in df.columns:
            x = pd.to_numeric(df[kol], errors="coerce")
            mask = (x < lo) | (x > hi)
            n = int(mask.sum())
            df.loc[mask, kol] = np.nan
            n_total += n
    print(f"    [3] Koreksi antropometri: {n_total} nilai mustahil -> NaN")
    return df

def _zscore_lms(value, L, M, S):
    value = np.asarray(value, dtype=float)
    if L == 0:
        return np.log(value / M) / S
    return (np.power(value / M, L) - 1.0) / (L * S)

def muat_referensi_who(dir_ref=DIR_REF_WHO):
    import zscore_who
    return zscore_who.muat_referensi_who()

def harmoniskan_posisi(df, source):
    peta = REKODE_POSISI_PER_SUMBER.get(source)
    if peta and "measure_position" in df.columns:
        x = pd.to_numeric(df["measure_position"], errors="coerce")
        df["measure_position"] = x.map(peta).fillna(x)
        print(f"    [3b] measure_position diselaraskan ke kanonik (tukar 1<->2) utk {source}")
    return df

def _rekode_kode_ke_kode(s, peta):
    x = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    for kode_src, kode_dst in peta.items():
        if kode_dst is not None:
            out[x == kode_src] = float(kode_dst)
    return out

def harmoniskan_kategorik_per_sumber(df, source):
    n = 0
    for tabel in (REKODE_NOMINAL_PER_SUMBER, REKODE_ORDINAL_PER_SUMBER):
        for fitur, peta in tabel.get(source, {}).items():
            if fitur in df.columns:
                df[fitur] = _rekode_kode_ke_kode(df[fitur], peta)
                n += 1
    if n:
        print(f"    [5b] {n} fitur nominal/ordinal diselaraskan ke kode kanonik SSGI 2022 utk {source}")
    return df

def tahap4_zscore(df, ref_who=None):
    import zscore_who
    if ref_who is None:
        ref_who = zscore_who.muat_referensi_who()

    PETUNJUK = {
        "height_child_cm": ["tinggi", "height", "panjang", "length", "tb", "1503", "j02"],
        "weight_child_kg": ["berat", "weight", "bb", "1502", "1509", "j01"],
        "age_child_months": ["umur", "age", "bln", "month", "usia", "4072", "b4k7"],
        "sex_child": ["sex", "kelamin", "jenis", "jk", "404", "b4k4"],
    }
    hilang = [c for c in PETUNJUK if c not in df.columns]
    if hilang:
        print(f"    [4] ! KOLOM WAJIB Z-SCORE TIDAK TERPETAKAN: {hilang}")
        for c in hilang:
            kand = [k for k in df.columns
                    if any(t in k.lower() for t in PETUNJUK[c])][:10]
            print(f"        '{c}' tidak ada. Kandidat kolom serupa di data: {kand}")
        raise ValueError(
            f"Z-score batal: kolom wajib {hilang} tidak terpetakan untuk sumber ini. "
            f"Cocokkan kode kamus K dengan kandidat di atas (atau lihat baris "
            f"'tak tercocokkan' pada log tahap 2), lalu jalankan ulang.")

    kol_posisi = "measure_position" if "measure_position" in df.columns else None

    df = zscore_who.hitung_zscore(
        df, ref_who,
        kol_tinggi="height_child_cm", kol_berat="weight_child_kg",
        kol_umur="age_child_months", kol_sex="sex_child",
        kol_posisi=kol_posisi,
        flag_haz=RANGE_ZSCORE["haz_score"],
        flag_waz=RANGE_ZSCORE["waz_score"],
        flag_whz=RANGE_ZSCORE["whz_score"],
    )
    n_stunting = int(pd.to_numeric(df["stunting_binary"], errors="coerce").sum())
    n_valid = int(pd.to_numeric(df["haz_score"], errors="coerce").notna().sum())
    print(f"    [4] Z-score WHO: HAZ valid={n_valid:,} | stunting={n_stunting:,} "
          f"(posisi: {'dipakai' if kol_posisi else 'tdk ada'})")
    return df

def tahap5_komposit(df, K):
    aset = [c for c in kelompok_kolom(K, prefix="asset_") if c in df.columns]
    if len(aset) >= 3:
        df["wealth_index"] = _wealth_index_pca(df[aset])
    print(f"    [5] wealth_index: PCA atas {len(aset)} aset")

    anc = [c for c in kelompok_kolom(K, prefix="anc_freq_") if c in df.columns]
    if anc:
        df["ANC_total_visit"] = df[anc].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    print(f"    [5] ANC_total_visit: jumlah {len(anc)} kolom frekuensi ANC")

    food = [c for c in kelompok_kolom(K, prefix="food_") if c in df.columns]
    if food:
        biner = df[food].apply(pd.to_numeric, errors="coerce")
        skor = (biner == 1).sum(axis=1)
        skor = skor.where(biner.notna().any(axis=1))
        df["food_diversity_score"] = skor
    print(f"    [5] food_diversity_score: dari {len(food)} kelompok pangan (baduta-only)")

    if "ttd_count" in df.columns:
        c = pd.to_numeric(df["ttd_count"], errors="coerce")
        df["TTD_compliance"] = (c >= 90).astype("Int64")
        df.loc[c.isna(), "TTD_compliance"] = pd.NA
    print("    [5] TTD_compliance: ambang >=90 tablet")

    imm = [c for c in kelompok_kolom(K, prefix="imm_") if c in df.columns]
    imm = [c for c in imm if c != "imm_vit_a_count"]
    if imm:
        biner = df[imm].apply(pd.to_numeric, errors="coerce")
        df["imunisasi_lengkap"] = (biner == 1).all(axis=1).astype("Int64")
    print(f"    [5] imunisasi_lengkap: dari {len(imm)} jenis imunisasi dasar")
    return df

def _wealth_index_pca(df_aset):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X = df_aset.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean())
    Xs = StandardScaler().fit_transform(X)
    pc1 = PCA(n_components=1).fit_transform(Xs).ravel()
    if np.corrcoef(pc1, X.sum(axis=1))[0, 1] < 0:
        pc1 = -pc1
    return pc1

def _ke_biner(s):
    x = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    out[x == 1] = 1.0
    out[(x == 0) | (x == 2)] = 0.0
    return out.astype("Int8")

def _rekode_biner_kustom(s, nilai_1, nilai_0):
    x = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    out[x.isin(nilai_1)] = 1.0
    out[x.isin(nilai_0)] = 0.0
    return out.astype("Int8")

def tahap6_encoding(df, source):
    drop = [c for c in SKEMA_ENCODING["drop"] if c in df.columns]
    df = df.drop(columns=drop, errors="ignore")

    kol_ganti = [c for c in df.columns if c not in SKEMA_ENCODING["meta"]]
    df[kol_ganti] = df[kol_ganti].replace(list(KODE_MISSING_KHUSUS), np.nan)

    rekode_sumber = REKODE_BINER_PER_SUMBER.get(source, {})
    n_bin = 0
    for c in SKEMA_ENCODING["biner"]:
        if c in df.columns:
            if c in rekode_sumber:
                v1, v0 = rekode_sumber[c]
                df[c] = _rekode_biner_kustom(df[c], v1, v0)
            elif c in REKODE_BINER_KHUSUS:
                v1, v0 = REKODE_BINER_KHUSUS[c]
                df[c] = _rekode_biner_kustom(df[c], v1, v0)
            else:
                df[c] = _ke_biner(df[c])
            n_bin += 1

    n_num = 0
    for jenis in ("ordinal", "nominal", "kontinu"):
        for c in SKEMA_ENCODING[jenis]:
            if c in df.columns and df[c].dtype == "object":
                df[c] = pd.to_numeric(df[c], errors="coerce")
                n_num += 1

    print(f"    [6] Encoding: {n_bin} biner->0/1 | nominal disimpan sbg kode "
          f"(One-Hot saat modeling) | ordinal/kontinu numerik")
    return df

def simpan_skema_encoding(path=None):
    import json
    path = path or os.path.join(DIR_OUTPUT, "skema_encoding.json")
    os.makedirs(DIR_OUTPUT, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(SKEMA_ENCODING, f, indent=2, ensure_ascii=False)
    print(f"  Skema encoding ditulis: {path}")

def tahap7_mice(df, source):
    import miceforest as mf
    import gc

    if "kohort" not in df.columns:
        print(f"    [7] MICE ({source}): kohort tidak ada -> dilewati")
        return df

    biner_set = set(SKEMA_ENCODING["biner"])
    nominal_set = set(SKEMA_ENCODING["nominal"])
    hasil_kohort = []
    ringkas = []

    for koh, sub in df.groupby("kohort", dropna=False):
        sub = sub.copy()
        if pd.isna(koh):
            hasil_kohort.append(sub)
            continue

        kandidat = [c for c in sub.columns if c not in MICE_KECUALIKAN
                    and not c.endswith("_missing")]
        fitur = [c for c in kandidat if c not in MICE_JANGAN_IMPUTASI]

        nan_penuh = [c for c in fitur if sub[c].isna().all()]
        fitur = [c for c in fitur if c not in nan_penuh]

        tak_layak = []
        for c in fitur:
            non_na = pd.to_numeric(sub[c], errors="coerce").dropna()
            if len(non_na) < MICE_MIN_TERISI or non_na.nunique() < 2:
                tak_layak.append(c)
        if tak_layak:
            fitur = [c for c in fitur if c not in tak_layak]
            print(f"        ! kolom tak layak MICE (data minim/variasi nol), dilewati: {tak_layak}")

        obj_cols = [c for c in fitur if sub[c].dtype == "object"]
        if obj_cols:
            print(f"        ! kolom object (dipaksa numerik): {obj_cols}")

        for c in kandidat:
            rate = sub[c].isna().mean()
            if (rate >= MICE_AMBANG_INDIKATOR or c in MICE_JANGAN_IMPUTASI) and rate > 0:
                sub[f"{c}_missing"] = sub[c].isna().astype("Int8")

        perlu = [c for c in fitur if sub[c].isna().any()]
        if not perlu:
            hasil_kohort.append(sub)
            ringkas.append(f"{koh}: 0 fitur diimputasi")
            continue

        idx_asli = sub.index
        matriks = sub[fitur].reset_index(drop=True).copy()
        kategori_kode = (biner_set | nominal_set | set(SKEMA_ENCODING["ordinal"]))
        nilai_valid = {}
        for c in fitur:
            matriks[c] = pd.to_numeric(matriks[c], errors="coerce")
            if c in kategori_kode:
                vals = np.sort(matriks[c].dropna().unique())
                nilai_valid[c] = vals

        fitur_impute = list(matriks.columns)

        kernel = mf.ImputationKernel(matriks, num_datasets=MICE_NUM_DATASETS,
                                     mean_match_candidates=0,
                                     random_state=MICE_RANDOM_STATE)
        kernel.mice(iterations=MICE_ITERATIONS,
                    min_data_in_leaf=MICE_MIN_DATA_IN_LEAF,
                    max_bin=MICE_MAX_BIN, force_col_wise=True)
        terisi = kernel.complete_data(dataset=0)
        terisi.index = idx_asli

        for c in fitur_impute:
            val = pd.to_numeric(terisi[c], errors="coerce")
            if c in nilai_valid and len(nilai_valid[c]) > 0:
                arr = nilai_valid[c]
                idx = np.abs(val.to_numpy()[:, None] - arr[None, :]).argmin(axis=1)
                val = pd.Series(arr[idx], index=val.index)
            sub[c] = val

        hasil_kohort.append(sub)
        ringkas.append(f"{koh}: {len(perlu)} fitur diimputasi ({len(nan_penuh)} NaN-penuh dilewati)")

        del kernel, terisi, matriks
        gc.collect()

    out = pd.concat(hasil_kohort, axis=0).sort_index()
    print(f"    [7] MICE ({source}): " + " | ".join(ringkas))
    return out

def tahap8_gabung(list_df_source):
    frames = []
    for source, df in list_df_source.items():
        d = df.copy()
        d["source_flag"] = source
        frames.append(d)
    gab = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    ind_cols = [c for c in gab.columns if c.endswith("_missing")]
    for ic in ind_cols:
        fitur = ic[:-len("_missing")]
        if fitur in gab.columns:
            perlu = gab[ic].isna()
            if perlu.any():
                gab.loc[perlu, ic] = 0
        gab[ic] = gab[ic].astype("Int8")

    print(f"    [8] Concat: {len(frames)} sumber -> {gab.shape[0]} baris, "
          f"{gab.shape[1]} kolom (source_flag + {len(ind_cols)} indikator missing)")
    return gab

def tahap9_validasi(df):
    laporan = []
    laporan.append(("total_baris", len(df)))
    laporan.append(("total_kolom", df.shape[1]))
    if "source_flag" in df.columns:
        laporan.append(("baris_per_sumber", df["source_flag"].value_counts().to_dict()))
    if "kohort" in df.columns:
        laporan.append(("baris_per_kohort", df["kohort"].value_counts(dropna=False).to_dict()))
    if "stunting_binary" in df.columns and "kohort" in df.columns:
        prev = df.groupby("kohort")["stunting_binary"].apply(
            lambda x: round(float(pd.to_numeric(x, errors="coerce").mean()), 4)).to_dict()
        laporan.append(("prevalensi_stunting_per_kohort", prev))

    print("    [9] Validasi:")
    for k, v in laporan:
        print(f"         - {k}: {v}")

    diagnosa_missing_struktural(df)

    for c in df.columns:
        if df[c].dtype == "object":
            num = pd.to_numeric(df[c], errors="coerce")
            if num.notna().mean() >= 0.95:
                df[c] = num
            else:
                df[c] = df[c].astype("string")

    os.makedirs(DIR_OUTPUT, exist_ok=True)
    df.to_parquet(PARQUET_FINAL, index=False)
    print(f"    [9] Parquet anonim ditulis: {PARQUET_FINAL}")
    print("        (HANYA file ini yang boleh naik ke Colab untuk training ML)")
    return df

def proses_satu_sumber(source, K, SOURCE_COL, ref_who=None):
    print(f"\n  === Sumber: {source.upper()} ===")
    df, meta = baca_sav(PATHS[source])
    print(f"    Dibaca: {df.shape[0]} baris, {df.shape[1]} kolom")

    df = tahap1_anonimisasi(df, source, K, SOURCE_COL)
    df = tahap2_pemetaan(df, source, K, SOURCE_COL)
    df = tahap3_koreksi_antropometri(df)
    df = tambah_kohort(df)
    df = harmoniskan_posisi(df, source)
    df = tahap4_zscore(df, ref_who)
    df = tahap5_komposit(df, K)
    df = saring_kolom_standar(df, K)
    df = harmoniskan_kategorik_per_sumber(df, source)
    df = tahap6_encoding(df, source)
    df = tahap7_mice(df, source)
    return df

def main():
    print("=" * 70)
    print(" PIPELINE HARMONISASI STUNTING: 9 TAHAP")
    print(f" Mulai: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    K, SOURCE_COL = muat_kamus()
    print(f"  Kamus dimuat: {len(K)} variabel standar")

    ref_who = muat_referensi_who()
    print("  Referensi WHO LMS dimuat (lhfa/wfa/wfl/wfh)")

    hasil = {}
    for source in PATHS:
        hasil[source] = proses_satu_sumber(source, K, SOURCE_COL, ref_who)

    gabung = tahap8_gabung(hasil)
    final = tahap9_validasi(gabung)
    simpan_skema_encoding()

    print("\n" + "=" * 70)
    print(f" SELESAI: {final.shape[0]} baris x {final.shape[1]} kolom")
    print("=" * 70)

if __name__ == "__main__":
    main()
