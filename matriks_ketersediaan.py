import os
import importlib.util

import numpy as np
import pandas as pd

PIPELINE = "harmonisasi_pipeline.py"
OUT_CSV = "output_harmonisasi/matriks_ketersediaan.csv"

AMBANG_TERSEDIA = 0.50
AMBANG_ABSEN = 0.05

def muat_pipeline(path=PIPELINE):
    spec = importlib.util.spec_from_file_location("pipe", path)
    pipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipe)
    return pipe

def label(pct):
    if pct >= AMBANG_TERSEDIA:
        return "tersedia"
    if pct < AMBANG_ABSEN:
        return "absen"
    return "parsial"

def hitung_ketersediaan(pipe):
    K, SOURCE_COL = pipe.muat_kamus()
    kecuali = set(pipe.MICE_KECUALIKAN) | set(pipe.SKEMA_ENCODING["drop"])
    nama_standar = [r[0] for r in K if r[7] != "META" and r[0] not in kecuali]
    set_fitur = set(nama_standar)
    kategori = {r[0]: r[1] for r in K if r[0] in set_fitur}

    sel = {}
    n_sel = {}

    for source in pipe.PATHS:
        print(f"  Memproses {source} ...")
        df, _ = pipe.baca_sav(pipe.PATHS[source])
        df = pipe.tahap1_anonimisasi(df, source, K, SOURCE_COL)
        df = pipe.tahap2_pemetaan(df, source, K, SOURCE_COL)

        umur = pd.to_numeric(df.get("age_child_months"), errors="coerce")
        kohort = np.where(umur < pipe.AMBANG_BADUTA, "baduta", "balita_tua")

        kol_ganti = [c for c in df.columns if c not in pipe.SKEMA_ENCODING["meta"]]
        df[kol_ganti] = df[kol_ganti].replace(list(pipe.KODE_MISSING_KHUSUS), np.nan)

        for koh in ("baduta", "balita_tua"):
            mask = pd.Series(kohort == koh, index=df.index)
            n = int(mask.sum())
            n_sel[(source, koh)] = n
            for v in nama_standar:
                if v in df.columns and n > 0:
                    kol = df[v]
                    if isinstance(kol, pd.DataFrame):
                        kol = kol.iloc[:, 0]
                    pct = float(kol[mask.values].notna().mean())
                else:
                    pct = 0.0
                sel[(v, source, koh)] = pct

    return nama_standar, kategori, sel, n_sel

def bangun_matriks(nama_standar, kategori, sel, sumber_list):
    kohort_list = ["baduta", "balita_tua"]
    baris = []
    for v in nama_standar:
        row = {"variabel": v, "kategori": kategori.get(v, "")}
        tersedia_baduta = 0
        tersedia_tua = 0
        for s in sumber_list:
            for koh in kohort_list:
                pct = sel.get((v, s, koh), 0.0)
                row[f"{s}_{koh}"] = round(pct, 3)
                if koh == "baduta" and pct >= AMBANG_TERSEDIA:
                    tersedia_baduta += 1
                if koh == "balita_tua" and pct >= AMBANG_TERSEDIA:
                    tersedia_tua += 1
        row["n_sumber_baduta"] = tersedia_baduta
        row["n_sumber_balita_tua"] = tersedia_tua
        row["lintas_sumber_baduta"] = ("kuat" if tersedia_baduta == 3
                                       else "sebagian" if tersedia_baduta == 2
                                       else "lemah" if tersedia_baduta == 1
                                       else "tidak")
        baris.append(row)
    return pd.DataFrame(baris)

def ringkas(mat, sumber_list):
    print("\n" + "=" * 68)
    print(" RINGKASAN KETERSEDIAAN")
    print("=" * 68)

    kuat = mat[mat["n_sumber_baduta"] == 3]["variabel"].tolist()
    print(f"\n Tersedia di KETIGA sumber (baduta), {len(kuat)} variabel:")
    print("   " + ", ".join(kuat[:40]) + (" ..." if len(kuat) > 40 else ""))

    heterogen = mat[(mat["n_sumber_baduta"].isin([1, 2]))][
        ["variabel", "kategori", "n_sumber_baduta"]]
    print(f"\n Heterogen antar-sumber (baduta), {len(heterogen)} variabel "
          f"(HATI-HATI utk klaim lintas-sumber):")
    for _, r in heterogen.iterrows():
        print(f"   {r['variabel']:24} [{r['kategori']:22}] tersedia di {r['n_sumber_baduta']}/3 sumber")

    baduta_only = mat[(mat["n_sumber_baduta"] >= 1) & (mat["n_sumber_balita_tua"] == 0)][
        "variabel"].tolist()
    print(f"\n Baduta-only (absen di balita_tua semua sumber), {len(baduta_only)} variabel:")
    print("   " + ", ".join(baduta_only))

def main():
    print("=" * 68)
    print(" MEMBANGUN MATRIKS KETERSEDIAAN VARIABEL")
    print("=" * 68)
    pipe = muat_pipeline()
    sumber_list = list(pipe.PATHS.keys())

    nama_standar, kategori, sel, n_sel = hitung_ketersediaan(pipe)
    mat = bangun_matriks(nama_standar, kategori, sel, sumber_list)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    mat.to_csv(OUT_CSV, index=False)
    print(f"\n Matriks ditulis: {OUT_CSV}")
    print(" Jumlah baris per sel:")
    for (s, k), n in n_sel.items():
        print(f"   {s}-{k}: {n:,}")

    ringkas(mat, sumber_list)

if __name__ == "__main__":
    main()
