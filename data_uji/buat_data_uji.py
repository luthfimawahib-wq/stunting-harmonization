from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

DIR_SKRIP = os.path.dirname(os.path.abspath(__file__))
AKAR_REPO = os.path.abspath(os.path.join(DIR_SKRIP, os.pardir))
OUT_DIR = os.path.join(AKAR_REPO, "output_harmonisasi")
PATH_ASLI_DEFAULT = os.path.join(OUT_DIR, "stunting_harmonized.parquet")
PATH_SKEMA_DEFAULT = os.path.join(OUT_DIR, "skema_encoding.json")
PATH_MATRIKS_DEFAULT = os.path.join(OUT_DIR, "matriks_ketersediaan.csv")
PATH_KONTRAK_DEFAULT = os.path.join(OUT_DIR, "kontrak_skema_uji.json")
PATH_OUT_DEFAULT = os.path.join(OUT_DIR, "stunting_harmonized.parquet")

SUMBER = ["ssgi22", "ssgi24", "ski23"]
KOHORT = ["baduta", "balita_tua"]
PREVALENSI_DEFAULT = {"baduta": 0.1427, "balita_tua": 0.2215}
PROP_SUMBER = {"ssgi22": 0.464, "ssgi24": 0.416, "ski23": 0.120}
PROP_BADUTA = 0.36
META_NAMA = {"source_flag", "kohort", "province", "district",
             "svy_weight", "svy_psu", "svy_strata", "id_ruta"}
TARGET = "stunting_binary"

SPEK_KONTINU = {
    "age_child_months": (30, 16, 0, 59), "height_child_cm": (85, 12, 45, 125),
    "weight_child_kg": (11.5, 3.0, 2, 30), "lila_child_cm": (14.5, 1.8, 8, 22),
    "haz_score": (-1.0, 1.3, -6, 6), "waz_score": (-0.9, 1.2, -6, 5),
    "whz_score": (-0.4, 1.2, -5, 5), "weight_gain_trend": (0.0, 1.0, -3, 3),
    "birth_weight_g": (3100, 480, 1000, 5000), "birth_length_cm": (48, 2.6, 38, 56),
    "gestational_age_wk": (38.5, 1.8, 28, 43), "head_circ_birth_cm": (34, 1.8, 28, 40),
    "age_mother_yr": (29, 6.2, 15, 49), "gravida": (2.2, 1.2, 1, 8),
    "height_mother_cm": (153, 5.8, 135, 175), "weight_mother_kg": (56, 10, 35, 110),
    "lila_mother_cm": (26, 3.0, 17, 40),
    "anc_freq_doc_t1": (1.0, 1.0, 0, 6), "anc_freq_doc_t2": (1.2, 1.1, 0, 6),
    "anc_freq_doc_t3": (2.0, 1.4, 0, 8), "anc_freq_mid_t1": (1.1, 1.0, 0, 6),
    "anc_freq_mid_t2": (1.3, 1.1, 0, 6), "anc_freq_mid_t3": (2.1, 1.5, 0, 8),
    "ttd_count": (55, 30, 0, 120), "imd_duration": (35, 25, 0, 180),
    "weaning_age_months": (13, 6, 0, 24), "mpasi_age_months": (6.2, 1.6, 0, 18),
    "meal_frequency": (3.0, 1.1, 1, 8), "imm_vit_a_count": (1.4, 0.9, 0, 4),
    "weigh_freq_12mo": (7, 3.2, 0, 12), "water_fetch_time": (8, 12, 0, 120),
    "floor_area_m2": (60, 32, 8, 400), "hb_level_gdl": (11.2, 1.5, 5, 17),
    "retinol_level": (30, 9, 5, 70), "wealth_index": (0.0, 1.0, -4, 4),
    "ANC_total_visit": (6, 3.0, 0, 20), "food_diversity_score": (4.2, 1.8, 0, 8),
}

def muat_skema(path):
    if path and os.path.exists(path):
        sk = json.load(open(path, encoding="utf-8"))
        for k in ("biner", "ordinal", "nominal", "kontinu", "meta", "drop"):
            sk.setdefault(k, [])
        return sk
    return None

def sel_key(s, k):
    return f"{s}|{k}"

def kontrak_dari_asli(path_asli, skema):
    df = pd.read_parquet(path_asli)
    cols = list(df.columns)
    dtype = {c: str(df[c].dtype) for c in cols}

    kat = set()
    if skema:
        kat = (set(skema["biner"]) | set(skema["ordinal"]) | set(skema["nominal"]))
    kat.discard(TARGET)

    grp = df.groupby(["source_flag", "kohort"]) if {"source_flag", "kohort"} <= set(cols) else None
    cell_counts, struct_absent = {}, {c: [] for c in cols}
    allnan = [c for c in cols if df[c].isna().all()]
    if grp is not None:
        for (s, k), idx in grp.groups.items():
            cell_counts[sel_key(s, k)] = int(len(idx))
        for (s, k), sub in grp:
            na_all = sub.isna().all()
            for c in cols:
                if bool(na_all.get(c, False)):
                    struct_absent[c].append([s, k])

    domains = {}
    for c in cols:
        if c in kat and c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals):
                domains[c] = sorted(set(np.round(vals.to_numpy()).astype("int64").tolist()))
            else:
                domains[c] = []

    prevalensi = dict(PREVALENSI_DEFAULT)
    if {TARGET, "kohort"} <= set(cols):
        pv = df.groupby("kohort")[TARGET].mean()
        for k in KOHORT:
            if k in pv.index and pd.notna(pv[k]):
                prevalensi[k] = float(pv[k])

    src_vals = sorted(df["source_flag"].dropna().astype(str).unique().tolist()) if "source_flag" in cols else SUMBER
    koh_vals = sorted(df["kohort"].dropna().astype(str).unique().tolist()) if "kohort" in cols else KOHORT

    return {
        "columns": cols, "dtype": dtype, "domains": domains,
        "struct_absent": struct_absent, "allnan": allnan,
        "cell_counts": cell_counts, "prevalensi": prevalensi,
        "source_vals": src_vals, "kohort_vals": koh_vals,
    }

def kontrak_dari_skema_matriks(skema, matriks_path):
    nondrop = skema["biner"] + skema["ordinal"] + skema["nominal"] + skema["kontinu"] + skema["meta"]
    kecuali = {TARGET, "haz_score", "waz_score", "whz_score", "source_flag", "kohort",
               "province", "district", "svy_weight", "svy_psu", "svy_strata", "id_ruta"}
    kat = set(skema["biner"]) | set(skema["ordinal"]) | set(skema["nominal"])
    kat.discard(TARGET)
    struct_absent = {c: [] for c in nondrop}
    domains = {}
    if matriks_path and os.path.exists(matriks_path):
        mat = pd.read_csv(matriks_path)
        cells = [(s, k) for s in SUMBER for k in KOHORT]
        for _, r in mat.iterrows():
            c = str(r["variabel"])
            if c not in struct_absent:
                continue
            for s, k in cells:
                col = f"{s}_{k}"
                if col in mat.columns and float(r[col]) <= 0.02:
                    struct_absent[c].append([s, k])
    for c in kat:
        if c in skema["biner"]:
            domains[c] = [0, 1]
        elif c == "edu_mother":
            domains[c] = [1, 2, 3, 4, 5]
        else:
            domains[c] = list(range(1, 7))
    cols = nondrop[:]
    dtype = {}
    for c in cols:
        if c in skema["biner"]:
            dtype[c] = "Int8"
        elif c in ("source_flag", "kohort"):
            dtype[c] = "string"
        elif c in kecuali - {"source_flag", "kohort"} or c in skema["kontinu"] or c in skema["nominal"] or c in skema["ordinal"] or c in skema["meta"]:
            dtype[c] = "float64"
        else:
            dtype[c] = "float64"
    dtype[TARGET] = "Int8"
    cell_counts = {sel_key(s, k): int(round(1000 * PROP_SUMBER[s] * (PROP_BADUTA if k == "baduta" else 1 - PROP_BADUTA))) for s in SUMBER for k in KOHORT}
    return {"columns": cols, "dtype": dtype, "domains": domains, "struct_absent": struct_absent,
            "allnan": [], "cell_counts": cell_counts, "prevalensi": dict(PREVALENSI_DEFAULT),
            "source_vals": SUMBER, "kohort_vals": KOHORT}

def _kontinu(rng, n, nama):
    mean, sd, lo, hi = SPEK_KONTINU.get(nama, (0.0, 1.0, None, None))
    x = rng.normal(mean, sd, n)
    if lo is not None:
        x = np.clip(x, lo, hi)
    return x

def bangun(kontrak, skema, n_total, seed):
    rng = np.random.default_rng(seed)
    cols = kontrak["columns"]
    dtype = kontrak["dtype"]
    domains = kontrak["domains"]
    absent = {c: set(tuple(x) for x in v) for c, v in kontrak["struct_absent"].items()}
    allnan = set(kontrak["allnan"])
    prevalensi = kontrak["prevalensi"]
    kontinu_set = set(skema["kontinu"]) if skema else set(SPEK_KONTINU)

    cc = kontrak["cell_counts"]
    tot = sum(cc.values()) or 1
    baris = []
    for key, cnt in cc.items():
        s, k = key.split("|")
        nc = int(round(n_total * cnt / tot))
        baris += [(s, k)] * nc
    if not baris:
        for s in SUMBER:
            for k in KOHORT:
                baris += [(s, k)] * int(round(n_total * PROP_SUMBER[s] * (PROP_BADUTA if k == "baduta" else 1 - PROP_BADUTA)))
    rng.shuffle(baris)
    n = len(baris)
    sflag = np.array([b[0] for b in baris], dtype=object)
    koh = np.array([b[1] for b in baris], dtype=object)
    mask_sel = {(s, k): (sflag == s) & (koh == k) for s in SUMBER for k in KOHORT}

    data = {}
    base_cols = [c for c in cols if not c.endswith("_missing")]
    for c in base_cols:
        if c == "source_flag":
            data[c] = sflag
            continue
        if c == "kohort":
            data[c] = koh
            continue
        col = np.full(n, np.nan)
        if c in allnan:
            data[c] = col
            continue
        for s in SUMBER:
            for k in KOHORT:
                m = mask_sel[(s, k)]
                if not m.any() or (s, k) in absent.get(c, set()):
                    continue
                nn = int(m.sum())
                if c == TARGET:
                    p = prevalensi.get(k, 0.15)
                    col[m] = (rng.random(nn) < p).astype(float)
                elif c in domains and domains[c]:
                    col[m] = rng.choice(domains[c], nn).astype(float)
                elif c in kontinu_set:
                    col[m] = _kontinu(rng, nn, c)
                elif c == "svy_weight":
                    col[m] = np.clip(rng.lognormal(0.0, 0.4, nn), 0.05, 12)
                elif c == "id_ruta":
                    base = {"ssgi22": 1_000_000, "ssgi24": 2_000_000, "ski23": 3_000_000}[s]
                    col[m] = base + rng.integers(0, max(1, int(nn / 1.6)), nn)
                elif c in ("svy_psu", "province", "svy_strata", "district"):
                    hi = {"svy_psu": 400, "province": 100, "svy_strata": 40, "district": 100}[c]
                    col[m] = rng.integers(1, hi, nn)
                else:
                    col[m] = rng.integers(0, 2, nn)
        data[c] = col

    for c in cols:
        if not c.endswith("_missing"):
            continue
        dasar = c[:-len("_missing")]
        if dasar in data:
            data[c] = np.isnan(pd.to_numeric(pd.Series(data[dasar]), errors="coerce").to_numpy()).astype(float)
        else:
            data[c] = np.zeros(n)

    df = pd.DataFrame(data)[cols]

    for c in cols:
        dt = dtype.get(c, "float64")
        s = df[c]
        if dt in ("int8", "int16", "int32", "int64") and s.isna().any():
            dt = dt.capitalize()
        try:
            df[c] = s.astype(dt)
        except (ValueError, TypeError):
            try:
                df[c] = s.astype(dt.capitalize())
            except Exception:
                pass
    return df

def main(argv=None):
    ap = argparse.ArgumentParser(description="Pembangkit Parquet sintetis berskema sama persis.")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--asli", default=PATH_ASLI_DEFAULT, help="Parquet asli utk mengklon skema")
    ap.add_argument("--skema", default=PATH_SKEMA_DEFAULT)
    ap.add_argument("--matriks", default=PATH_MATRIKS_DEFAULT)
    ap.add_argument("--kontrak", default=PATH_KONTRAK_DEFAULT)
    ap.add_argument("--out", default=PATH_OUT_DEFAULT)
    args = ap.parse_args(argv)

    skema = muat_skema(args.skema)

    if os.path.exists(args.asli):
        kontrak = kontrak_dari_asli(args.asli, skema)
        sumber_kontrak = f"Parquet asli ({args.asli})"
        os.makedirs(os.path.dirname(os.path.abspath(args.kontrak)), exist_ok=True)
        json.dump(kontrak, open(args.kontrak, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    elif os.path.exists(args.kontrak):
        kontrak = json.load(open(args.kontrak, encoding="utf-8"))
        sumber_kontrak = f"kontrak JSON ({args.kontrak})"
    elif skema is not None:
        kontrak = kontrak_dari_skema_matriks(skema, args.matriks)
        sumber_kontrak = "fallback skema+matriks (APPROKSIMASI, tidak dijamin lolos validasi)"
    else:
        sys.exit("[GAGAL] tidak ada Parquet asli, kontrak JSON, maupun skema. "
                 "Beri --asli atau --skema.")

    df = bangun(kontrak, skema, args.n, args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_parquet(args.out, index=False)

    prev = df.groupby("kohort")[TARGET].mean().round(4).to_dict() if {TARGET, "kohort"} <= set(df.columns) else {}
    print(f"[OK] {args.out}")
    print(f"     sumber kontrak : {sumber_kontrak}")
    print(f"     baris x kolom  : {df.shape[0]} x {df.shape[1]}")
    print(f"     prevalensi     : {prev}")
    print(f"     CATATAN        : data SINTETIS; hubungan antar-variabel artifisial.")

if __name__ == "__main__":
    main()
