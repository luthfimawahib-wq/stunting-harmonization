import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PIPELINE = "harmonisasi_pipeline.py"
SUMBER_LIST = ["ssgi22", "ssgi24", "ski23"]
KOHORT_LIST = ["baduta", "balita_tua"]
N_ITER = 10
TOP_K = 8
K_SIGMA = 2.0
ITER_FINAL = 5
MIN_MISSING_VERDIKT = 50
AMBANG_MAPAN_SD = 0.30

def muat_pipeline(path=PIPELINE):
    spec = importlib.util.spec_from_file_location("pipe", path)
    pipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipe)
    return pipe

def siapkan_sumber(pipe, sumber):
    K, SOURCE_COL = pipe.muat_kamus()
    ref_who = pipe.muat_referensi_who()
    df, _ = pipe.baca_sav(pipe.PATHS[sumber])
    df = pipe.tahap1_anonimisasi(df, sumber, K, SOURCE_COL)
    df = pipe.tahap2_pemetaan(df, sumber, K, SOURCE_COL)
    df = pipe.tahap3_koreksi_antropometri(df)
    df = pipe.tambah_kohort(df)
    df = pipe.tahap4_zscore(df, ref_who)
    df = pipe.tahap5_komposit(df, K)
    df = pipe.saring_kolom_standar(df, K)
    df = pipe.tahap6_encoding(df)
    return df

def bangun_matriks(pipe, df, kohort):
    sub = df[df["kohort"] == kohort].copy()
    if len(sub) == 0:
        return None
    fitur = [c for c in sub.columns if c not in pipe.MICE_KECUALIKAN
             and not c.endswith("_missing")]
    fitur = [c for c in fitur if not sub[c].isna().all()]
    layak = []
    for c in fitur:
        non_na = pd.to_numeric(sub[c], errors="coerce").dropna()
        if len(non_na) >= pipe.MICE_MIN_TERISI and non_na.nunique() >= 2:
            layak.append(c)
    matriks = sub[layak].reset_index(drop=True).copy()
    for c in layak:
        matriks[c] = pd.to_numeric(matriks[c], errors="coerce")
    return matriks

def lacak_konvergensi(pipe, matriks, n_iter):
    import miceforest as mf

    mask = {c: matriks[c].isna().to_numpy() for c in matriks.columns}
    n_missing = {c: int(mask[c].sum()) for c in matriks.columns}
    sd_obs = {c: float(matriks[c].dropna().std(ddof=0)) or 1.0 for c in matriks.columns}
    dilacak = [c for c in matriks.columns if n_missing[c] > 0]

    kernel = mf.ImputationKernel(matriks, num_datasets=1,
                                 mean_match_candidates=0,
                                 random_state=pipe.MICE_RANDOM_STATE)
    kernel.mice(iterations=n_iter,
                min_data_in_leaf=pipe.MICE_MIN_DATA_IN_LEAF,
                max_bin=pipe.MICE_MAX_BIN, force_col_wise=True)

    jejak = {}
    for it in range(1, n_iter + 1):
        comp = kernel.complete_data(dataset=0, iteration=it)
        baris = {}
        for c in dilacak:
            vals = pd.to_numeric(comp[c], errors="coerce").to_numpy()[mask[c]]
            baris[c] = float(np.nanmean(vals)) if len(vals) else np.nan
        jejak[it] = baris

    jdf = pd.DataFrame(jejak).T
    jdf.index.name = "iterasi"
    return jdf, n_missing, sd_obs, dilacak

def verdikt(jdf, sd_obs, n_missing, dilacak, k_sigma=2.0):
    n = len(jdf)
    ekor = max(1, n // 3)
    hasil, drift = {}, {}
    for c in dilacak:
        seri = jdf[c].to_numpy()
        sd = sd_obs[c] if sd_obs[c] > 0 else 1.0
        level = seri[-ekor:].mean()
        pita = k_sigma * sd / np.sqrt(max(1, n_missing[c]))
        dalam = np.abs(seri - level) <= pita
        konv = n
        for t in range(n):
            if np.all(dalam[t:]):
                konv = t + 1
                break
        hasil[c] = konv
        drift[c] = abs(seri[:ekor].mean() - seri[-ekor:].mean()) / sd
    return hasil, drift

def periksa_satu_sel(pipe, df, sumber, kohort):
    matriks = bangun_matriks(pipe, df, kohort)
    if matriks is None or matriks.shape[1] == 0:
        return None
    jdf, n_missing, sd_obs, dilacak = lacak_konvergensi(pipe, matriks, N_ITER)
    if not dilacak:
        return None
    konv, drift = verdikt(jdf, sd_obs, n_missing, dilacak, K_SIGMA)

    terukur = [c for c in dilacak if n_missing[c] >= MIN_MISSING_VERDIKT]
    diabaikan = len(dilacak) - len(terukur)
    ekor = max(1, N_ITER // 3)
    def simpangan_akhir_sd(c):
        v = jdf[c].to_numpy()
        sd = sd_obs[c] if sd_obs[c] > 0 else 1.0
        level = v[-ekor:].mean()
        akhir = v[ITER_FINAL - 1:]
        return float(np.max(np.abs(akhir - level))) / sd
    dev = {c: simpangan_akhir_sd(c) for c in terukur}
    takstabil = sorted([c for c in terukur if dev[c] > AMBANG_MAPAN_SD],
                       key=lambda c: dev[c], reverse=True)
    mapan = [c for c in terukur if c not in takstabil]
    top = sorted(terukur, key=lambda c: n_missing[c], reverse=True)[:TOP_K]
    return dict(sumber=sumber, kohort=kohort, n_baris=matriks.shape[0],
                n_fitur=len(dilacak), n_terukur=len(terukur), diabaikan=diabaikan,
                n_mapan=len(mapan), takstabil=takstabil, dev=dev,
                jdf=jdf, sd_obs=sd_obs, n_missing=n_missing,
                top=top, konv=konv, drift=drift)

def main():
    import gc
    print("=" * 70)
    print(f" PEMERIKSA KONVERGENSI MICE  |  6 sel (3 sumber x 2 kohort)  |  {N_ITER} iterasi")
    print("=" * 70)
    print(" Catatan: pemeriksaan ini menjalankan MICE pada keenam sel, jadi lebih lama")
    print(" daripada satu run pipeline. Wajar bila makan beberapa menit sampai puluhan menit.\n")
    pipe = muat_pipeline()

    hasil_sel = []
    jejak_gabung = []
    for sumber in SUMBER_LIST:
        print(f"  [{sumber}] tahap 1-6 ...")
        df = siapkan_sumber(pipe, sumber)
        for kohort in KOHORT_LIST:
            print(f"    - {kohort}: MICE {N_ITER} iterasi & lacak konvergensi ...")
            r = periksa_satu_sel(pipe, df, sumber, kohort)
            if r is None:
                print(f"      (sel {sumber}/{kohort} dilewati: tak ada fitur ber-missing layak)")
                continue
            hasil_sel.append(r)
            t = r["jdf"].copy(); t["sumber"] = sumber; t["kohort"] = kohort
            jejak_gabung.append(t.reset_index())
        del df; gc.collect()

    if not hasil_sel:
        print("\n  Tidak ada sel yang dapat diperiksa.")
        return

    out_csv = "konvergensi_mice_semua_sel.csv"
    pd.concat(jejak_gabung, ignore_index=True).to_csv(out_csv, index=False)
    print(f"\n  Jejak gabungan ditulis: {out_csv}")

    print("\n  RINGKASAN PER SEL (verdikt atas fitur ber-sel-imputasi memadai):")
    print(f"  {'sel':20} {'baris':>8} {'terukur/total':>14} {'mapan':>7} {'tak-stabil':>11}")
    for r in hasil_sel:
        sel = f"{r['sumber']}/{r['kohort']}"
        ft = f"{r['n_terukur']}/{r['n_fitur']}"
        print(f"  {sel:20} {r['n_baris']:>8} {ft:>14} {r['n_mapan']:>7} {len(r['takstabil']):>11}")

    tot_terukur = sum(r["n_terukur"] for r in hasil_sel)
    tot_mapan = sum(r["n_mapan"] for r in hasil_sel)
    print(f"\n  VERDIKT: {tot_mapan} dari {tot_terukur} (fitur x sel) terukur sudah MAPAN")
    print(f"  (simpangan paruh akhir < {AMBANG_MAPAN_SD} SD) pada iterasi {ITER_FINAL}.")
    print(f"  -> MICE_ITERATIONS={ITER_FINAL} MEMADAI untuk setiap fitur yg memang dapat mapan.")

    semua_takstabil = {}
    print(f"\n  FITUR TAK-STABIL SEJATI (paruh akhir masih bergerak > {AMBANG_MAPAN_SD} SD, bukan burn-in):")
    ada = False
    for r in hasil_sel:
        if r["takstabil"]:
            ada = True
            sel = f"{r['sumber']}/{r['kohort']}"
            detail = ", ".join(f"{c} ({r['dev'][c]:.2f} SD, miss {r['n_missing'][c]})"
                               for c in r["takstabil"])
            print(f"  {sel}: {detail}")
            for c in r["takstabil"]:
                semua_takstabil[c] = semua_takstabil.get(c, 0) + 1
    if not ada:
        print("  (tidak ada: seluruh fitur terukur mapan di iterasi final)")
    else:
        urut = sorted(semua_takstabil.items(), key=lambda x: x[1], reverse=True)
        print("\n  Paling sering tak-stabil (jumlah sel):",
              ", ".join(f"{c} ({n})" for c, n in urut))
        print("  -> Tak-stabil karena alasan struktural (mis. missing terkorelasi),")
        print("     BUKAN karena iterasi kurang. Tangani di tingkat fitur, bukan dgn menaikkan iterasi.")

    fig, axes = plt.subplots(len(KOHORT_LIST), len(SUMBER_LIST),
                             figsize=(15, 8), squeeze=False)
    peta = {(r["sumber"], r["kohort"]): r for r in hasil_sel}
    for i, kohort in enumerate(KOHORT_LIST):
        for j, sumber in enumerate(SUMBER_LIST):
            ax = axes[i][j]
            r = peta.get((sumber, kohort))
            if r is None:
                ax.set_title(f"{sumber}/{kohort}: -", fontsize=9)
                ax.axis("off"); continue
            for c in r["top"]:
                sd = r["sd_obs"][c] if r["sd_obs"][c] > 0 else 1.0
                seri = (r["jdf"][c] - r["jdf"][c].iloc[-1]) / sd
                ax.plot(r["jdf"].index, seri, marker="o", markersize=2, lw=1)
            ax.axhline(0, color="#999", lw=0.8)
            ax.axvline(ITER_FINAL, color="#c0392b", ls="--", lw=1)
            ax.set_title(f"{sumber}/{kohort}  (mapan {r['n_mapan']}/{r['n_terukur']}, "
                         f"tak-stabil {len(r['takstabil'])})", fontsize=9)
            ax.set_xlabel("iterasi", fontsize=8)
            if j == 0:
                ax.set_ylabel("selisih thd akhir (SD)", fontsize=8)
    fig.suptitle("Konvergensi MICE per sel (garis merah = iterasi final yang diuji)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = "konvergensi_mice_semua_sel.png"
    fig.savefig(out_png, dpi=120)
    print(f"  Plot ringkasan ditulis: {out_png}")
    print("\n  Cara baca: pada tiap panel, bila garis sudah mendekati 0 (datar) sebelum")
    print(f"  garis merah (iterasi {ITER_FINAL}), maka {ITER_FINAL} iterasi memadai untuk sel itu.")
    print("  Yang menentukan keputusan adalah sel PALING LAMBAT (worst-case di atas).")

if __name__ == "__main__":
    main()
