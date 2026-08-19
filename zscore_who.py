#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 MODUL Z-SCORE WHO 2006 (metode LMS) — untuk pipeline harmonisasi stunting
================================================================================
 Menghitung HAZ (TB/U), WAZ (BB/U), WHZ (BB/TB) berdasarkan WHO Child Growth
 Standards 2006, metode LMS:  Z = ((X/M)^L - 1) / (L*S),  atau ln(X/M)/S bila L=0.

 Referensi LMS resmi WHO diambil dari paket `pygrowup` (tabel JSON bawaan),
 TETAPI seluruh perhitungan ditulis di sini agar transparan & dapat diaudit —
 tidak bergantung pada API pygrowup.

 Penanganan kritis yang sesuai pedoman WHO:
   1. Koreksi panjang<->tinggi:
        - <24 bln diukur BERDIRI  -> panjang = tinggi + 0.7 cm
        - >=24 bln diukur TELENTANG-> tinggi  = panjang - 0.7 cm
   2. WHZ memakai tabel berbeda: weight-for-LENGTH (<24bln, 45-110cm) vs
      weight-for-HEIGHT (>=24bln, 65-120cm).
   3. Flagging WHO (nilai mustahil biologis -> NaN):
        HAZ < -6 atau > 6 ; WAZ < -6 atau > 5 ; WHZ < -5 atau > 5
   4. stunting_binary = 1 bila HAZ < -2.

 Catatan: lookup berbasis umur-BULAN (standar utk data survei). WHO idealnya
 berbasis umur-hari; selisihnya kecil utk klasifikasi <-2 SD dan dapat dijelaskan
 di limitasi.
================================================================================
"""

import os
import json
import bisect

import numpy as np
import pandas as pd


# ==============================================================================
# MUAT TABEL LMS WHO (dari pygrowup) -> struktur lookup numerik
# ==============================================================================
def _dir_tabel():
    import pygrowup
    return os.path.join(os.path.dirname(pygrowup.__file__), "tables")


def _muat_lms(nama_file, kunci_indeks):
    """Baca satu tabel JSON -> (indeks_terurut, dict{idx: (L,M,S)})."""
    path = os.path.join(_dir_tabel(), nama_file)
    data = json.load(open(path))
    tabel = {}
    for row in data:
        idx = float(row[kunci_indeks])
        tabel[idx] = (float(row["L"]), float(row["M"]), float(row["S"]))
    idx_terurut = sorted(tabel.keys())
    return idx_terurut, tabel


def muat_referensi_who():
    """Muat semua tabel LMS yg dibutuhkan, per jenis kelamin.
    Return: dict ref[indikator][sex] = (idx_terurut, {idx:(L,M,S)})
    indikator: 'haz_l' (length-based 0-24bln), 'haz_h' (height-based 24-60bln),
               'waz' (per bulan),
               'whz_l' (weight-for-length, per cm), 'whz_h' (weight-for-height, per cm)
    sex: 1=laki-laki, 2=perempuan (konvensi SSGI/SKI)
    """
    ref = {"haz_l": {}, "haz_h": {}, "waz": {}, "whz_l": {}, "whz_h": {}}
    files = {
        # HAZ length-based (0-24 bln, diukur telentang) vs height-based (24-60 bln, berdiri)
        ("haz_l", 1): ("lhfa_boys_0_2_zscores.json", "Month"),
        ("haz_l", 2): ("lhfa_girls_0_2_zscores.json", "Month"),
        ("haz_h", 1): ("lhfa_boys_2_5_zscores.json", "Month"),
        ("haz_h", 2): ("lhfa_girls_2_5_zscores.json", "Month"),
        ("waz", 1): ("wfa_boys_0_5_zscores.json", "Month"),
        ("waz", 2): ("wfa_girls_0_5_zscores.json", "Month"),
        ("whz_l", 1): ("wfl_boys_0_2_zscores.json", "Length"),
        ("whz_l", 2): ("wfl_girls_0_2_zscores.json", "Length"),
        ("whz_h", 1): ("wfh_boys_2_5_zscores.json", "Height"),
        ("whz_h", 2): ("wfh_girls_2_5_zscores.json", "Height"),
    }
    for (indikator, sex), (fn, kunci) in files.items():
        ref[indikator][sex] = _muat_lms(fn, kunci)
    return ref


# ==============================================================================
# LOOKUP LMS + RUMUS Z-SCORE
# ==============================================================================
def _ambil_lms(idx_terurut, tabel, x):
    """Ambil (L,M,S) utk nilai indeks x. Interpolasi linear antar titik terdekat.
    Di luar rentang tabel -> None (akan jadi NaN)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if x < idx_terurut[0] or x > idx_terurut[-1]:
        return None
    if x in tabel:
        return tabel[x]
    # interpolasi linear antara dua titik tetangga
    pos = bisect.bisect_left(idx_terurut, x)
    lo, hi = idx_terurut[pos - 1], idx_terurut[pos]
    L0, M0, S0 = tabel[lo]
    L1, M1, S1 = tabel[hi]
    w = (x - lo) / (hi - lo)
    return (L0 + w * (L1 - L0), M0 + w * (M1 - M0), S0 + w * (S1 - S0))


def _z_lms(value, lms):
    """Z = ((X/M)^L - 1)/(L*S), atau ln(X/M)/S bila L=0."""
    if lms is None or value is None or np.isnan(value) or value <= 0:
        return np.nan
    L, M, S = lms
    if L == 0:
        return np.log(value / M) / S
    return (np.power(value / M, L) - 1.0) / (L * S)


# ==============================================================================
# KOREKSI PANJANG <-> TINGGI
# ==============================================================================
def _tinggi_untuk_haz(tinggi, umur_bln, posisi_telentang):
    """Konsistenkan utk lhfa: <24bln pakai PANJANG (telentang), >=24bln pakai TINGGI.
    posisi_telentang: True jika diukur telentang (recumbent), False jika berdiri,
                      None jika tak diketahui (asumsikan sesuai umur)."""
    if np.isnan(tinggi) or np.isnan(umur_bln):
        return tinggi
    if umur_bln < 24:
        # standar: panjang (telentang). Jika diukur BERDIRI -> +0.7 utk jadikan panjang.
        if posisi_telentang == False:
            return tinggi + 0.7
        return tinggi
    else:
        # standar: tinggi (berdiri). Jika diukur TELENTANG -> -0.7 utk jadikan tinggi.
        if posisi_telentang == True:
            return tinggi - 0.7
        return tinggi


# ==============================================================================
# FUNGSI UTAMA: hitung semua Z-score utk satu DataFrame
# ==============================================================================
def hitung_zscore(df, ref,
                  kol_tinggi="height_child_cm", kol_berat="weight_child_kg",
                  kol_umur="age_child_months", kol_sex="sex_child",
                  kol_posisi=None,
                  flag_haz=(-6, 6), flag_waz=(-6, 5), flag_whz=(-5, 5)):
    """Tambahkan kolom haz_score, waz_score, whz_score, stunting_binary ke df.
    kol_posisi: nama kolom posisi pengukuran (opsional). Konvensi diasumsikan
                1=telentang, 2=berdiri (sesuaikan bila perlu)."""
    tinggi = pd.to_numeric(df.get(kol_tinggi), errors="coerce").to_numpy(dtype=float)
    berat  = pd.to_numeric(df.get(kol_berat),  errors="coerce").to_numpy(dtype=float)
    umur   = pd.to_numeric(df.get(kol_umur),   errors="coerce").to_numpy(dtype=float)
    sex    = pd.to_numeric(df.get(kol_sex),    errors="coerce").to_numpy(dtype=float)
    if kol_posisi and kol_posisi in df.columns:
        pos_raw = pd.to_numeric(df[kol_posisi], errors="coerce").to_numpy(dtype=float)
    else:
        pos_raw = np.full(len(df), np.nan)

    n = len(df)
    haz = np.full(n, np.nan)
    waz = np.full(n, np.nan)
    whz = np.full(n, np.nan)

    for i in range(n):
        s = sex[i]
        if s not in (1, 2):
            continue
        u = umur[i]

        # --- HAZ (pilih tabel length<24bln / height>=24bln + koreksi posisi) ---
        posisi_telentang = (None if np.isnan(pos_raw[i])
                            else bool(pos_raw[i] == 1))  # bool() hindari numpy.bool_
        tb_haz = _tinggi_untuk_haz(tinggi[i], u, posisi_telentang)
        if not np.isnan(u):
            kunci_haz = "haz_l" if u < 24 else "haz_h"
            idxs, tab = ref[kunci_haz][int(s)]
            haz[i] = _z_lms(tb_haz, _ambil_lms(idxs, tab, u))

        # --- WAZ ---
        if not np.isnan(u):
            idxs, tab = ref["waz"][int(s)]
            waz[i] = _z_lms(berat[i], _ambil_lms(idxs, tab, u))

        # --- WHZ (pilih tabel berdasarkan umur: <24 length, >=24 height) ---
        if not np.isnan(tinggi[i]) and not np.isnan(u):
            kunci = "whz_l" if u < 24 else "whz_h"
            idxs, tab = ref[kunci][int(s)]
            whz[i] = _z_lms(berat[i], _ambil_lms(idxs, tab, tinggi[i]))

    # flagging WHO (mustahil biologis -> NaN)
    haz = np.where((haz < flag_haz[0]) | (haz > flag_haz[1]), np.nan, haz)
    waz = np.where((waz < flag_waz[0]) | (waz > flag_waz[1]), np.nan, waz)
    whz = np.where((whz < flag_whz[0]) | (whz > flag_whz[1]), np.nan, whz)

    df = df.copy()
    df["haz_score"] = haz
    df["waz_score"] = waz
    df["whz_score"] = whz
    df["stunting_binary"] = pd.array(np.where(np.isnan(haz), np.nan, (haz < -2).astype(float)),
                                     dtype="Int64")
    return df


# ==============================================================================
# UJI MANDIRI: nilai median harus -> Z = 0
# ==============================================================================
if __name__ == "__main__":
    ref = muat_referensi_who()

    # Anak laki-laki 24 bulan pada median tinggi WHO: M lhfa boys @24bln (height-based)
    idxs, tab = ref["haz_h"][1]
    L, M, S = tab[24.0]
    print(f"Median TB laki-laki 24 bln (WHO M, height) = {M:.2f} cm")

    contoh = pd.DataFrame({
        "height_child_cm": [M, M - 2*1.0, 80.0],   # baris1 = tepat median
        "weight_child_kg": [12.0, 12.0, 10.0],
        "age_child_months": [24, 24, 24],
        "sex_child": [1, 1, 2],
    })
    hasil = hitung_zscore(contoh, ref)
    print(hasil[["haz_score", "waz_score", "whz_score", "stunting_binary"]].round(3))
    print(f"\nUji: HAZ baris median = {hasil['haz_score'].iloc[0]:.4f} (harus ~0)")
    assert abs(hasil["haz_score"].iloc[0]) < 1e-6, "GAGAL: median harus Z=0"
    print("LULUS: nilai median menghasilkan HAZ = 0.")
