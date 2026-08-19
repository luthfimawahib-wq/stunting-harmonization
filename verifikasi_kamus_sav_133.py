import os
import re
import sys
import difflib
import pandas as pd

try:
    import pyreadstat
except ImportError:
    print("ERROR: pyreadstat belum terpasang. Jalankan: pip install pyreadstat pandas")
    sys.exit(1)

FILES = {
    "ssgi22": "raw_data/SSGI_2022.sav",
    "ssgi24": "raw_data/SSGI_2024.sav",
    "ski23":  "raw_data/SKI_2023.sav",
}
OUTPUT_DIR = "hasil_verifikasi"
LABEL_SIMILARITY_THRESHOLD = 0.45

PII_KEYWORDS = [
    "nik", "nama", "alamat", "no hp", "nomor hp", "telepon", "telpon",
    "rt", "rw", "kk", "kartu keluarga", "nomor urut", "responden",
    "pewawancara", "enumerator", "gps", "koordinat", "lintang", "bujur",
    "id art", "id ruta", "id rumah tangga", "tanggal lahir", "tgl lahir",
]

K = [
("stunting_binary","LABEL / OUTPUT","Status Stunting (TB/U klasifikasi)","DERIVASI P1503C+P4072+P404","DERIVASI P1512+P4072+P404","DERIVASI J02B+b4k7bln+B4K4","TB/U (teks)","LABEL"),
("haz_score","LABEL / OUTPUT","Skor Z TB menurut umur (HAZ)","DERIVASI P1503C+P4072+P404","DERIVASI P1512+P4072+P404","DERIVASI J02B+b4k7bln+B4K4","ZS TB/U","LABEL"),
("waz_score","LABEL / OUTPUT","Skor Z BB menurut umur (WAZ)","DERIVASI P1502C+P4072+P404","DERIVASI P1509+P4072+P404","DERIVASI J01C+b4k7bln+B4K4","ZS BB/U","LABEL"),
("whz_score","LABEL / OUTPUT","Skor Z BB menurut TB (WHZ/wasting)","DERIVASI P1502C+P1503C","DERIVASI P1509+P1512","DERIVASI J01C+J02B","ZS BB/TB","LABEL"),
("id_ruta","IDENTITAS & WILAYAH","ID Rumah Tangga (kunci klaster grouped-CV; ID internal dataset Kemenkes, bukan PII)","Cases_ruta","IDRUTA","IDRT","No","META"),
("pii_identitas","IDENTITAS & WILAYAH","NIK/Nama/Alamat/RT/RW","—","—","—","NIK/Nama/Alamat","HAPUS_PRIVASI"),
("province","IDENTITAS & WILAYAH","Provinsi","P101","P101","B1R1","Prov","IDENTITAS_WILAYAH"),
("district","IDENTITAS & WILAYAH","Kabupaten/Kota","P102","P102","B1R2","Kab/Kota","IDENTITAS_WILAYAH"),
("subdistrict","IDENTITAS & WILAYAH","Kecamatan","P103","—","—","Kec","IDENTITAS_WILAYAH"),
("village","IDENTITAS & WILAYAH","Desa/Kelurahan","P104","—","—","Desa/Kel","IDENTITAS_WILAYAH"),
("health_facility","IDENTITAS & WILAYAH","Puskesmas/Posyandu","—","—","—","Puskesmas/Posyandu","IDENTITAS_WILAYAH"),
("area_type","IDENTITAS & WILAYAH","Klasifikasi Desa (Kota/Desa)","P105","P105","B1R5","—","FITUR_UTAMA"),
("sex_child","DEMOGRAFI BALITA","Jenis Kelamin Balita","P404","P404","B4K4","JK","FITUR_UTAMA"),
("child_birth_date","DEMOGRAFI BALITA","Tanggal Lahir Balita","P406","P406","tgl_lahir","Tgl Lahir","FITUR_PENDUKUNG"),
("age_child_months","DEMOGRAFI BALITA","Umur Balita (bulan)","UMUR_BL_B","P4072","b4k7bln","Usia Saat Ukur","FITUR_UTAMA"),
("measure_date","DEMOGRAFI BALITA","Tanggal Pengukuran Antropometri","—","—","—","Tanggal Pengukuran","HAPUS_PRIVASI"),
("weight_child_kg","ANTROPOMETRI BALITA","BB Balita Saat Ukur (kg)","P1502C","P1509","J01C","Berat","FITUR_UTAMA"),
("height_child_cm","ANTROPOMETRI BALITA","TB Balita Saat Ukur (cm)","P1503C","P1512","J02B","Tinggi","FITUR_UTAMA"),
("measure_position","ANTROPOMETRI BALITA","Posisi Ukur (Berdiri/Telentang)","P1503D","P1511","J02C","Cara Ukur","FITUR_PENDUKUNG"),
("lila_child_cm","ANTROPOMETRI BALITA","LiLA Balita (cm)","—","P1514","—","LiLA","FITUR_PENDUKUNG"),
("weight_gain_trend","ANTROPOMETRI BALITA","Naik BB (tren pertumbuhan)","—","—","—","Naik Berat Badan","FITUR_PENDUKUNG"),
("birth_weight_g","DATA KELAHIRAN","Berat Badan Lahir (gram)","P1405","P915FK3","I05A","BB Lahir","FITUR_UTAMA"),
("birth_length_cm","DATA KELAHIRAN","Panjang Badan Lahir (cm)","P1408","P915GK3","I07","TB Lahir","FITUR_UTAMA"),
("gestational_age_wk","DATA KELAHIRAN","Usia Kehamilan Saat Lahir (minggu)","P1401","P915EK3","I04","—","FITUR_UTAMA"),
("delivery_place","DATA KELAHIRAN","Tempat Persalinan","P1402","—","H29","—","FITUR_PENDUKUNG"),
("delivery_assistant","DATA KELAHIRAN","Penolong Persalinan","P1403","—","H28","—","FITUR_PENDUKUNG"),
("delivery_method","DATA KELAHIRAN","Metode Persalinan (Normal/Sesar)","—","—","H32","—","FITUR_PENDUKUNG"),
("head_circ_birth_cm","DATA KELAHIRAN","Lingkar Kepala Lahir (cm)","P1411","—","I10","—","FITUR_PENDUKUNG"),
("edu_mother","DEMOGRAFI & STATUS IBU","Pendidikan Ibu","DIDIK_IBU","P409_IBU","DIDIK_IBU","—","FITUR_UTAMA"),
("occupation_mother","DEMOGRAFI & STATUS IBU","Pekerjaan Ibu","KERJA_IBU","P410","KERJA_IBU","—","FITUR_PENDUKUNG"),
("age_mother_yr","DEMOGRAFI & STATUS IBU","Usia Ibu (tahun)","UMUR_IBU","P4073","UMUR_IBU","—","FITUR_PENDUKUNG"),
("marital_status","DEMOGRAFI & STATUS IBU","Status Kawin Ibu","P405","P405_IBU","B4K5","—","FITUR_PENDUKUNG"),
("gravida","DEMOGRAFI & STATUS IBU","Urutan Kehamilan/Gravida","—","P901","H02A","—","FITUR_PENDUKUNG"),
("height_mother_cm","ANTROPOMETRI IBU","Tinggi Badan Ibu (cm)","P1211","P1516","—","—","FITUR_UTAMA"),
("weight_mother_kg","ANTROPOMETRI IBU","BB Ibu saat Hamil (kg)","P1210","P1503","—","—","FITUR_PENDUKUNG"),
("lila_mother_cm","ANTROPOMETRI IBU","LiLA Ibu (cm), KEK","P1802C (WUS)","—","J07B","—","FITUR_UTAMA"),
("anc_received","ANC & KEHAMILAN","Pernah ANC (Ya/Tidak)","P1201","—","H09","—","FITUR_UTAMA"),
("anc_freq_doc_t1","ANC & KEHAMILAN","Freq ANC Dokter Trimester 1","P1204A2","P909AK3","H11A2","—","FITUR_UTAMA"),
("anc_freq_doc_t2","ANC & KEHAMILAN","Freq ANC Dokter Trimester 2","P1204A3","P909AK4","H11A3","—","FITUR_UTAMA"),
("anc_freq_doc_t3","ANC & KEHAMILAN","Freq ANC Dokter Trimester 3","P1204A4","P909AK5","H11A4","—","FITUR_UTAMA"),
("anc_freq_mid_t1","ANC & KEHAMILAN","Freq ANC Bidan Trimester 1","P1204C2","P909CK3","H11C2","—","FITUR_UTAMA"),
("anc_freq_mid_t2","ANC & KEHAMILAN","Freq ANC Bidan Trimester 2","P1204C3","P909CK4","H11C3","—","FITUR_UTAMA"),
("anc_freq_mid_t3","ANC & KEHAMILAN","Freq ANC Bidan Trimester 3","P1204C4","P909CK5","H11C4","—","FITUR_UTAMA"),
("anc_place","ANC & KEHAMILAN","Tempat ANC Paling Sering","P1205","P910","H12","—","FITUR_PENDUKUNG"),
("anc_lila_measured","ANC & KEHAMILAN","LiLA Diukur Saat ANC","P1206D1","—","H13D","—","FITUR_PENDUKUNG"),
("anc_hb_tested","ANC & KEHAMILAN","Tes Hb Saat ANC","P1206N1","—","H13O","—","FITUR_PENDUKUNG"),
("pregnancy_class","ANC & KEHAMILAN","Kelas Ibu Hamil (ikut)","—","P904","H14","—","FITUR_PENDUKUNG"),
("kek_flag","KOMPLIKASI KEHAMILAN","Risiko KEK saat Hamil (LiLA<23.5)","—","P916D","—","—","FITUR_UTAMA"),
("anemia_preg_flag","KOMPLIKASI KEHAMILAN","Anemia saat Hamil (Hb<11)","—","P916C","—","—","FITUR_UTAMA"),
("hypertension_preg","KOMPLIKASI KEHAMILAN","Hipertensi/Pre-eklampsia saat Hamil","—","DERIVASI P916A+P916B","H22","—","FITUR_PENDUKUNG"),
("diabetes_preg","KOMPLIKASI KEHAMILAN","Diabetes saat Hamil","—","P916F","—","—","FITUR_PENDUKUNG"),
("ttd_received","SUPLEMENTASI IBU HAMIL","Mendapat TTD saat Hamil","P1207","P913","H18A","—","FITUR_UTAMA"),
("ttd_count","SUPLEMENTASI IBU HAMIL","Jumlah TTD Diminum (tablet)","P1208A3","P915DK3","H19A3","—","FITUR_UTAMA"),
("pmt_received","SUPLEMENTASI IBU HAMIL","Mendapat PMT saat Hamil","P1212","—","H56","—","FITUR_PENDUKUNG"),
("mms_received","SUPLEMENTASI IBU HAMIL","MMS (Gizi Mikro Multiple)","—","P914","—","—","FITUR_PENDUKUNG"),
("imd_skin_contact","PERSALINAN & IMD","IMD: Diletakkan di Dada Ibu","P1301A","—","I31A","—","FITUR_PENDUKUNG"),
("imd_duration","PERSALINAN & IMD","IMD: Lama Pelekatan Kulit","P1301C","—","I31C","—","FITUR_PENDUKUNG"),
("colostrum_action","ASI & PEMBERIAN MAKAN","Kolostrum: Tindakan Ibu","P1302","—","I33","—","FITUR_PENDUKUNG"),
("breastfed_ever","ASI & PEMBERIAN MAKAN","Pernah Diberi ASI","P1303","—","I35","—","FITUR_UTAMA"),
("breastfed_current","ASI & PEMBERIAN MAKAN","Masih Disusui Saat Ini","P1305","P1601","I37","—","FITUR_UTAMA"),
("weaning_age_months","ASI & PEMBERIAN MAKAN","Usia Disapih (bulan)","P1306","—","I38","—","FITUR_UTAMA"),
("prelacteal_feed","ASI & PEMBERIAN MAKAN","Makanan Sebelum ASI Pertama","P1308","—","I40","—","FITUR_PENDUKUNG"),
("mpasi_age_months","ASI & PEMBERIAN MAKAN","Usia Pertama MPASI (bulan)","P1311","—","—","—","FITUR_UTAMA"),
("food_water","KONSUMSI PANGAN 24 JAM","Air Putih","P1315A","P1603AK2","I50A","—","FITUR_PENDUKUNG"),
("food_formula","KONSUMSI PANGAN 24 JAM","Susu Formula","P1315D","P1603TK2","I50C","—","FITUR_PENDUKUNG"),
("food_cereal","KONSUMSI PANGAN 24 JAM","Serealia/Nasi/Bubur/Roti","P1315I","P1603BK2","I50J","—","FITUR_UTAMA"),
("food_vit_a_veg","KONSUMSI PANGAN 24 JAM","Labu/Wortel/Ubi Oranye","P1315J","P1603JK2","I50K","—","FITUR_UTAMA"),
("food_green_veg","KONSUMSI PANGAN 24 JAM","Sayuran Hijau","P1315L","P1603FK2","I50M","—","FITUR_UTAMA"),
("food_vit_a_fruit","KONSUMSI PANGAN 24 JAM","Buah Kaya Vitamin A","P1315M","P1603HK2","I50N","—","FITUR_UTAMA"),
("food_organ_meat","KONSUMSI PANGAN 24 JAM","Jeroan (Hati/Ampela)","P1315O","P1603QK2","I50P","—","FITUR_UTAMA"),
("food_meat","KONSUMSI PANGAN 24 JAM","Daging (Ayam/Sapi/dll)","P1315P","—","I50Q","—","FITUR_UTAMA"),
("food_egg","KONSUMSI PANGAN 24 JAM","Telur","P1315Q","P1603SK2","I50R","—","FITUR_UTAMA"),
("food_fish","KONSUMSI PANGAN 24 JAM","Ikan/Kerang","P1315R","P1603LK2","I50S","—","FITUR_UTAMA"),
("food_legume","KONSUMSI PANGAN 24 JAM","Kacang (Tahu/Tempe/Kedelai)","P1315S","P1603DK2","I50T","—","FITUR_PENDUKUNG"),
("meal_frequency","KONSUMSI PANGAN 24 JAM","Frekuensi Makan Padat/Hari","P1316A","—","I52","—","FITUR_UTAMA"),
("imm_hepb0","IMUNISASI","Hepatitis B-0 (neonatal)","P1005A","P1002AK2","I20A2","—","FITUR_UTAMA"),
("imm_bcg","IMUNISASI","BCG (1 bulan)","P1005B","P1002BK2","I20B2","—","FITUR_UTAMA"),
("imm_dpt1","IMUNISASI","DPT-HB-Hib 1 (2 bulan)","P1005C","P1002CK2","I20C2","—","FITUR_UTAMA"),
("imm_dpt2","IMUNISASI","DPT-HB-Hib 2 (3 bulan)","P1005D","P1002DK2","I20D2","—","FITUR_UTAMA"),
("imm_dpt3","IMUNISASI","DPT-HB-Hib 3 (4 bulan)","P1005E","P1002EK2","I20E2","—","FITUR_UTAMA"),
("imm_dpt_boost","IMUNISASI","DPT-HB-Hib Lanjutan (18 bln)","P1005F","P1002FK2","I20F2","—","FITUR_UTAMA"),
("imm_pcv","IMUNISASI","PCV 1,2,3","—","DERIVASI P1002GK2+P1002HK2+P1002IK2","DERIVASI I20G2+I20H2+I20I2","—","FITUR_PENDUKUNG"),
("imm_polio","IMUNISASI","Polio 1,2,3,4 (OPV/IPV)","DERIVASI P1005G+P1005H+P1005I+P1005J","—","DERIVASI I21A2+I21B2+I21C2+I21D2","—","FITUR_UTAMA"),
("imm_measles_9mo","IMUNISASI","Campak-Rubella/MR (9 bulan)","P1005N","P1002JK2","I20J2","—","FITUR_UTAMA"),
("imm_measles_boost","IMUNISASI","Campak-Rubella Lanjutan (18 bln)","P1005O","P1002KK2","I20K2","—","FITUR_UTAMA"),
("imm_vit_a_count","IMUNISASI","Jumlah Vitamin A Diterima","P1007G1","—","I29","Jml Vit A","FITUR_UTAMA"),
("weigh_freq_12mo","PEMANTAUAN TUMBUH KEMBANG","Freq Penimbangan BB 12 Bulan","P1007A3","—","I24","Naik Berat Badan*","FITUR_PENDUKUNG"),
("height_measure_12mo","PEMANTAUAN TUMBUH KEMBANG","Pengukuran TB/PB 12 Bulan","P1007B1","—","I26","—","FITUR_PENDUKUNG"),
("kpsp_sdidtk","PEMANTAUAN TUMBUH KEMBANG","Perkembangan KPSP/SDIDTK","—","—","DERIVASI I30A+I30B+I30C+I30D+I30E+I30F+I30G+I30H+I30I","KPSP","FITUR_PENDUKUNG"),
("kia_book","PEMANTAUAN TUMBUH KEMBANG","Kepemilikan Buku KIA","P1004","P906","I01","KIA","FITUR_PENDUKUNG"),
("ispa_1month","PENYAKIT BALITA","ISPA (1 bulan terakhir)","P901","P801","A05","—","FITUR_UTAMA"),
("diarrhea_1month","PENYAKIT BALITA","Diare (1 bulan terakhir)","P903","P804","A01","—","FITUR_UTAMA"),
("pneumonia_1year","PENYAKIT BALITA","Pneumonia (1 tahun terakhir)","P905","P807","A09","—","FITUR_UTAMA"),
("tb_1year","PENYAKIT BALITA","TB Paru (1 tahun terakhir)","P907","P810","A12","—","FITUR_PENDUKUNG"),
("worm_1year","PENYAKIT BALITA","Kecacingan (1 tahun terakhir)","P912","—","—","—","FITUR_PENDUKUNG"),
("jkn_owned","JAMINAN KESEHATAN","Kepemilikan JKN/BPJS","P412","P411","B4K11","—","FITUR_PENDUKUNG"),
("jkn_used","JAMINAN KESEHATAN","Pemanfaatan JKN Setahun","P413","P412","—","—","FITUR_PENDUKUNG"),
("water_source","SANITASI & AIR BERSIH","Sumber Air Minum Utama","P501","P501","B6R1*","—","FITUR_UTAMA"),
("sanitation_own","SANITASI & AIR BERSIH","Kepemilikan Jamban","P504","P509","—","—","FITUR_UTAMA"),
("toilet_type","SANITASI & AIR BERSIH","Jenis Kloset","P505","P512","—","—","FITUR_UTAMA"),
("feces_disposal","SANITASI & AIR BERSIH","Pembuangan Tinja","P506","P513","—","—","FITUR_UTAMA"),
("water_fetch_time","SANITASI & AIR BERSIH","Waktu Tempuh ke Sumber Air","—","P504","—","—","FITUR_PENDUKUNG"),
("building_ownership","PERUMAHAN","Status Kepemilikan Bangunan","—","P601","B7R1","—","FITUR_PENDUKUNG"),
("floor_area_m2","PERUMAHAN","Luas Lantai Bangunan (m2)","—","P602","B7R2","—","FITUR_PENDUKUNG"),
("cooking_fuel","PERUMAHAN","Bahan Bakar Memasak","P602","P605","B7R5","—","FITUR_PENDUKUNG"),
("lighting_source","PERUMAHAN","Sumber Penerangan","—","P603","B7R3A","—","FITUR_PENDUKUNG"),
("asset_gas","ASET (PROXY SOSEK)","Tabung Gas >=5.5 kg","P601A","P606A","B7R4A","—","FITUR_UTAMA"),
("asset_washing_machine","ASET (PROXY SOSEK)","Mesin Cuci","P601B","P606B","B7R4B","—","FITUR_UTAMA"),
("asset_fridge","ASET (PROXY SOSEK)","Lemari Es","P601C","P606C","B7R4C","—","FITUR_UTAMA"),
("asset_phone","ASET (PROXY SOSEK)","Handphone/Smartphone","P601G","P606G","B7R4E","—","FITUR_PENDUKUNG"),
("asset_computer","ASET (PROXY SOSEK)","Komputer/Laptop","P601H","P606H","B7R4H","—","FITUR_PENDUKUNG"),
("asset_tv","ASET (PROXY SOSEK)","Televisi Layar Datar","P601N","P606N","B7R4I","—","FITUR_PENDUKUNG"),
("asset_motorcycle","ASET (PROXY SOSEK)","Sepeda Motor","P601J","P606J","B7R4J","—","FITUR_UTAMA"),
("asset_car","ASET (PROXY SOSEK)","Mobil","P601M","P606M","B7R4M","—","FITUR_PENDUKUNG"),
("asset_gold","ASET (PROXY SOSEK)","Emas/Perhiasan >=10 gram","P601I","P606I","B7R4N","—","FITUR_UTAMA"),
("asset_land","ASET (PROXY SOSEK)","Tanah/Lahan","P601O","P606O","B7R4O","—","FITUR_UTAMA"),
("asset_livestock","ASET (PROXY SOSEK)","Hewan Ternak Kaki Empat","—","P606P","B7R4P","—","FITUR_PENDUKUNG"),
("bansos_kks","BANTUAN SOSIAL","KKS/Kartu Perlindungan Sosial","P603","P607","B7R6","—","FITUR_PENDUKUNG"),
("bansos_pkh","BANTUAN SOSIAL","Program Keluarga Harapan (PKH)","P604A","P608A","B7R7A","—","FITUR_PENDUKUNG"),
("bansos_bpnt","BANTUAN SOSIAL","BPNT/Bantuan Sembako","P604B","P608B","B7R7B","—","FITUR_PENDUKUNG"),
("bansos_blt","BANTUAN SOSIAL","Bantuan Langsung Tunai (BLT)","P604C","P608C","B7R7C","—","FITUR_PENDUKUNG"),
("know_stunting","PENGETAHUAN STUNTING","Tahu tentang Stunting","—","P1401","—","—","FITUR_UTAMA"),
("know_info_source","PENGETAHUAN STUNTING","Sumber Informasi Stunting","—","DERIVASI P1403+P1404","—","—","FITUR_PENDUKUNG"),
("percep_hereditary","PENGETAHUAN STUNTING","Persepsi: Stunting Keturunan","—","P1405","—","—","FITUR_UTAMA"),
("percep_asi","PENGETAHUAN STUNTING","Persepsi: Cegah via ASI Eksklusif","—","P1413","—","—","FITUR_UTAMA"),
("percep_ttd","PENGETAHUAN STUNTING","Persepsi: TTD Cegah BBLR/Stunting","—","P1414","—","—","FITUR_UTAMA"),
("percep_obese_misc","PENGETAHUAN STUNTING","Miskonsepsi: Anak Gemuk Tdk Stunting","—","P1419","—","—","FITUR_UTAMA"),
("percep_cognition","PENGETAHUAN STUNTING","Persepsi: Dampak ke Kecerdasan","—","P1416","—","—","FITUR_PENDUKUNG"),
("percep_environment","PENGETAHUAN STUNTING","Persepsi: Lingkungan Kotor Berisiko","—","P1418","—","—","FITUR_PENDUKUNG"),
("hb_level_gdl","BIOMEDIS (SKI 2023)","Kadar Hemoglobin Darah (g/dL)","—","—","BM2C1","—","FITUR_UTAMA"),
("retinol_level","BIOMEDIS (SKI 2023)","Kadar Retinol (Vit A) Darah","—","—","HasilUji_Retinol","—","FITUR_UTAMA"),
("malaria_rdt","BIOMEDIS (SKI 2023)","Hasil Uji Malaria (RDT)","—","—","BM2B3","—","FITUR_PENDUKUNG"),
("immunity_serology","BIOMEDIS (SKI 2023)","Uji Imunitas Campak/Rubella/Difteri/Tetanus","—","—","DERIVASI kdDifteri1+kdRubella+kdMeasles+kdTetanus","—","FITUR_PENDUKUNG"),

("svy_weight","DESAIN SURVEI","Bobot/Penimbang Sampling (level balita/individu)","weight_balita","w_ind_final","w_final","—","META"),
("svy_psu","DESAIN SURVEI","Primary Sampling Unit (klaster, unik PER survei)","PSU","PSU","PSU","—","META"),
("svy_strata","DESAIN SURVEI","Strata Desain Sampling","STRATA","STRATA_MASKING","STRATA","—","META"),
]

COL = {"nama":0, "kategori":1, "konsep":2, "ssgi22":3, "ssgi24":4, "ski23":5, "eppgbm":6, "peran":7}
SOURCE_COL = {"ssgi22": 3, "ssgi24": 4, "ski23": 5}

EKSPEKTASI_VALUE_LABEL = {
    "sex_child": {
        "ssgi22": {1: "laki", 2: "perempuan"},
        "ssgi24": {1: "laki", 2: "perempuan"},
        "ski23": {1: "laki", 2: "empuan"},
    },
    "area_type": {
        "ssgi22": {1: "perkotaan", 2: "perdesaan"},
        "ssgi24": {1: "perkotaan", 2: "perdesaan"},
        "ski23": {1: "perkotaan", 2: "perdesaan"},
    },
    "occupation_mother": {
        "ssgi22": {1: "bekerja", 2: "bekerja"},
        "ssgi24": {1: "bekerja", 2: "sekolah", 3: "polri", 4: "pegawai", 5: "wiraswasta", 6: "petani", 7: "nelayan", 8: "pembantu", 9: "lainnya"},
        "ski23": {1: "bekerja", 2: "sekolah", 3: "polri", 4: "pegawai", 5: "wiraswasta", 6: "petani", 7: "nelayan", 8: "pembantu", 9: "lainnya"},
    },
    "edu_mother": {
        "ssgi22": {1: "sekolah", 2: "sd", 3: "smp", 4: "sma", 5: "pt"},
        "ssgi24": {1: "sekolah", 2: "paud", 3: "sd", 4: "sd", 5: "sltp", 6: "sma", 7: "diii", 8: "perguruan"},
        "ski23": {1: "sekolah", 2: "sd", 3: "sd", 4: "sltp", 5: "slta", 6: "d1", 7: "pt"},
    },
    "ttd_received": {
        "ssgi22": {1: "program", 2: "sendiri", 3: "program", 4: "tidak"},
        "ssgi24": {1: "ya", 2: "tidak", 8: "tidak tahu"},
        "ski23": {1: "ya", 2: "tidak"},
    },
    "jkn_used": {
        "ssgi22": {1: "jamkesda", 2: "swasta", 3: "swasta", 4: "dimanfaatkan", 8: "tidak tahu"},
        "ssgi24": {1: "ya", 2: "tidak", 8: "tidak tahu"},
    },
    "kia_book": {
        "ssgi22": {1: "menunjukkan", 2: "menunjukkan", 3: "hilang", 4: "tidak pernah memil"},
        "ssgi24": {1: "menunjukkan", 2: "menunjukkan", 3: "hilang", 4: "tidak pernah memil"},
        "ski23": {1: "menunjukkan", 2: "menunjukkan", 3: "menunjukkan", 4: "menunjukkan", 5: "menunjukkan", 6: "hilang", 7: "tidak pernah memil"},
    },
    "water_source": {
        "ssgi22": {1: "kemasan", 2: "ulang", 3: "ledeng", 4: "hydrant", 5: "terminal", 6: "penampungan", 7: "terlindung", 8: "terlindung", 9: "terlindung", 10: "terlindung", 11: "sumur", 12: "melalui", 13: "permukaan", 14: "lainnya"},
        "ssgi24": {1: "bermerek", 2: "ulang", 3: "ledeng", 4: "sumur", 5: "terlindung", 6: "terlindung", 7: "terlindung", 8: "terlindung", 9: "penampungan", 10: "permukaan", 11: "lainnya"},
        "ski23": {1: "kemasan", 2: "ulang", 3: "perpipaan", 4: "sumur", 5: "terlindung", 6: "terlindung", 7: "terlindung", 8: "terlindung", 9: "penampungan", 10: "permukaan", 11: "hidran", 12: "terminal", 13: "eceran"},
    },
    "cooking_fuel": {
        "ssgi22": {1: "listrik", 2: "lpg", 3: "alam", 4: "biogas", 5: "minyak", 6: "arang", 7: "bakar"},
        "ssgi24": {1: "memasak", 2: "bakar", 3: "batubara", 4: "minyak", 5: "biogas", 6: "lpg", 7: "listrik"},
        "ski23": {1: "listrik", 2: "lpg", 3: "biogas", 4: "minyak", 5: "briket", 6: "bakar", 7: "memasak"},
    },
    "delivery_place": {
        "ssgi22": {1: "pemerintah", 2: "bersalin", 3: "puskesmas", 4: "praktek", 5: "praktik", 6: "pustu", 7: "poskesdes", 8: "rumah", 9: "lainnya", 88: "tidak tahu"},
        "ski23": {1: "pemerintah", 2: "swasta", 3: "klinik", 4: "puskesmas", 5: "praktek", 6: "praktek", 7: "poskesdes", 8: "rumah", 9: "lainnya"},
    },
    "jkn_owned": {
        "ssgi22": {1: "jamkesda", 2: "jamkesda", 3: "swasta", 4: "swasta", 5: "tidak punya", 8: "tidak tahu"},
        "ssgi24": {1: "bpjs", 2: "bpjs", 3: "bpjs", 4: "jamkesda", 5: "jamkesda", 6: "jamkesda", 7: "jamkesda", 8: "kesehatan", 9: "asuransi", 10: "asuransi", 11: "asuransi", 12: "jamkesda", 13: "jamkesda", 14: "jamkesda", 15: "jamkesda", 16: "lainnya", 17: "lainnya", 18: "lainnya", 19: "lainnya", 20: "jamkesda", 21: "jamkesda", 22: "jamkesda", 23: "jamkesda", 24: "asuransi", 25: "asuransi", 26: "asuransi", 27: "asuransi", 28: "jamkesda", 29: "jamkesda", 30: "jamkesda", 31: "jamkesda", 32: "tidak memiliki"},
        "ski23": {1: "bpjs", 2: "bpjs", 4: "jamkesda", 5: "jamkesda", 8: "asuransi", 10: "asuransi", 16: "lainnya", 32: "tidak ada", 99: "kombinasi"},
    },
    "toilet_type": {
        "ssgi22": {1: "leher", 2: "plengsengan", 3: "plengsengan", 4: "cemplung", 5: "cemplung", 6: "lainnya"},
        "ssgi24": {1: "leher", 2: "plengsengan", 3: "plengsengan", 4: "cemplung", 5: "lainnya"},
    },
    "feces_disposal": {
        "ssgi22": {1: "ipal", 2: "septic", 3: "cubluk", 4: "bertutup", 5: "bertutup", 6: "selokan", 7: "lapang", 8: "lainnya"},
        "ssgi24": {1: "ipal", 2: "septic", 3: "cubluk", 4: "bertutup", 5: "bertutup", 6: "sungai", 7: "lapang", 8: "lainnya"},
    },
    "anc_place": {
        "ssgi22": {1: "pemerintah", 2: "bersalin", 3: "puskesmas", 4: "pustu", 5: "praktik", 6: "poskesdes", 7: "posyandu", 8: "rumah"},
        "ssgi24": {1: "pemerintah", 2: "bersalin", 3: "puskesmas", 4: "puskesmas", 5: "praktik", 6: "poskesdes", 7: "posyandu", 8: "rumah"},
        "ski23": {1: "pemerintah", 2: "bersalin", 3: "puskesmas", 4: "pustu", 5: "praktik", 6: "poskesdes", 7: "posyandu", 8: "rumah"},
    },
    "anc_received": {
        "ssgi22": {1: "ya", 2: "tidak", 3: "tidak tahu"},
        "ski23": {1: "ya", 2: "tidak"},
    },
    "imd_skin_contact": {
        "ssgi22": {1: "ya", 2: "tidak", 8: "tidak tahu"},
        "ski23": {1: "ya", 2: "tidak", 3: "tidak tahu"},
    },
    "measure_position": {
        "ssgi22": {1: "berdiri", 2: "telentang"},
        "ssgi24": {1: "telentang", 2: "berdiri"},
        "ski23": {1: "berdiri", 2: "telentang"},
    },
}

def normalize(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def token_set_ratio(a, b):
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    ta, tb = set(na.split()), set(nb.split())
    if not ta or not tb:
        return 0.0
    jaccard = len(ta & tb) / len(ta | tb)
    seq = difflib.SequenceMatcher(None, na, nb).ratio()
    return max(jaccard, seq)

def parse_kode(cell):
    cell = (cell or "").strip()
    perlu_konfirmasi = "*" in cell
    cell_clean = cell.replace("*", "")

    if cell_clean == "—" or cell_clean == "":
        return {"status": "tidak_tersedia", "kode_utama": None,
                "komponen": [], "perlu_konfirmasi": False}

    if cell_clean.upper().startswith("DERIVASI"):
        sisa = cell_clean[len("DERIVASI"):].strip()
        komponen = re.split(r"[+\s]+", sisa) if sisa else []
        komponen = [k for k in komponen
                    if re.search(r"[A-Za-z].*\d|\d.*[A-Za-z]|[a-z]{3,}", k)]
        return {"status": "derivasi", "kode_utama": None,
                "komponen": komponen, "perlu_konfirmasi": perlu_konfirmasi}

    kode_utama = re.split(r"\s*[(/]", cell_clean)[0].strip()
    komponen = re.findall(r"[A-Za-z][A-Za-z0-9_]+", kode_utama)
    return {"status": "tersedia", "kode_utama": kode_utama,
            "komponen": komponen, "perlu_konfirmasi": perlu_konfirmasi}

def cari_kode_fleksibel(kode, kolom_tersedia):
    if kode is None:
        return None
    if kode in kolom_tersedia:
        return kode
    lower_map = {c.lower(): c for c in kolom_tersedia}
    if kode.lower() in lower_map:
        return lower_map[kode.lower()]
    strip_map = {re.sub(r"[_\s]", "", c.lower()): c for c in kolom_tersedia}
    return strip_map.get(re.sub(r"[_\s]", "", kode.lower()))

def baca_metadata(path):
    df, meta = pyreadstat.read_sav(path, metadataonly=True, apply_value_formats=False)
    return meta

def verifikasi_sumber(sumber, meta):
    kolom_tersedia = list(meta.column_names)
    label_map = dict(meta.column_names_to_labels or {})
    value_labels = dict(meta.variable_value_labels or {})
    idx = SOURCE_COL[sumber]
    hasil = []

    for baris in K:
        nama, kategori, konsep, peran = baris[0], baris[1], baris[2], baris[COL["peran"]]
        cell = baris[idx]
        parsed = parse_kode(cell)

        rec = {
            "no": len(hasil) + 1,
            "kategori": kategori,
            "nama_standar": nama,
            "konsep": konsep,
            "peran": peran,
            "kode_kamus": cell,
            "kode_eppgbm": baris[COL["eppgbm"]],
            "kode_ditemukan": "",
            "label_aktual": "",
            "kemiripan_label": "",
            "value_label_aktual": "",
            "status": "",
            "keterangan": "",
        }

        if parsed["status"] == "tidak_tersedia":
            rec["status"] = "TIDAK TERSEDIA (sesuai kamus)"
            rec["keterangan"] = "Tangani dengan missing_indicator_flag jika di-merge"
            hasil.append(rec); continue

        if parsed["status"] == "derivasi":
            ada, hilang = [], []
            for k in parsed["komponen"]:
                f = cari_kode_fleksibel(k, kolom_tersedia)
                (ada if f else hilang).append(f or k)
            if not parsed["komponen"]:
                rec["status"] = "DERIVASI (tanpa komponen kode)"
                rec["keterangan"] = "Dibuat di pipeline; tidak ada kode sumber langsung"
            elif not hilang:
                rec["status"] = "DERIVASI - semua komponen OK"
                rec["kode_ditemukan"] = ", ".join(ada)
            else:
                rec["status"] = "DERIVASI - komponen HILANG"
                rec["kode_ditemukan"] = ", ".join(ada)
                rec["keterangan"] = "Hilang: " + ", ".join(hilang)
            hasil.append(rec); continue

        kode_aktual = cari_kode_fleksibel(parsed["kode_utama"], kolom_tersedia)
        if kode_aktual is None:
            rec["status"] = "TIDAK DITEMUKAN"
            rec["keterangan"] = f"Kode '{parsed['kode_utama']}' tidak ada di file. Cek codebook."
            hasil.append(rec); continue

        rec["kode_ditemukan"] = kode_aktual
        label_aktual = label_map.get(kode_aktual, "")
        rec["label_aktual"] = label_aktual
        sim = token_set_ratio(konsep, label_aktual)
        rec["kemiripan_label"] = f"{sim:.2f}"

        if kode_aktual in value_labels:
            vl = value_labels[kode_aktual]
            rec["value_label_aktual"] = "; ".join(f"{k}={v}" for k, v in list(vl.items())[:6])

        catatan_vl = ""
        if nama in EKSPEKTASI_VALUE_LABEL:
            harapan = EKSPEKTASI_VALUE_LABEL[nama].get(sumber, {})
            vl_aktual = value_labels.get(kode_aktual, {})
            for kode_num, makna in harapan.items():
                aktual = normalize(vl_aktual.get(kode_num, ""))
                if makna not in aktual:
                    catatan_vl += f"[VL? {kode_num} harap '{makna}' dapat '{vl_aktual.get(kode_num,'-')}'] "

        if sim >= 0.70:
            rec["status"] = "COCOK"
        elif sim >= LABEL_SIMILARITY_THRESHOLD:
            rec["status"] = "COCOK (label agak beda)"
        else:
            rec["status"] = "PERLU TINJAU (label beda)"
            rec["keterangan"] = "Verifikasi manual: konsep vs label aktual"

        if parsed["perlu_konfirmasi"]:
            rec["keterangan"] = "[Kode bertanda *: konfirmasi codebook] " + rec["keterangan"]
        if catatan_vl:
            rec["keterangan"] = catatan_vl + rec["keterangan"]

        hasil.append(rec)

    return pd.DataFrame(hasil)

def deteksi_pii(meta):
    label_map = dict(meta.column_names_to_labels or {})
    rows = []
    for kode, label in label_map.items():
        token_label = set(normalize(label).split())
        token_kode = set(normalize(kode).split())
        nlabel = normalize(label)
        for kw in PII_KEYWORDS:
            kw_tokens = kw.split()
            cocok = (kw in nlabel) if len(kw_tokens) > 1 else (kw in token_label or kw in token_kode)
            if cocok:
                rows.append({"kode": kode, "label": label, "kata_kunci": kw}); break
    return pd.DataFrame(rows)

def dump_metadata_lengkap(meta):
    label_map = dict(meta.column_names_to_labels or {})
    value_labels = dict(meta.variable_value_labels or {})
    rows = []
    for kode in meta.column_names:
        vl = value_labels.get(kode, {})
        rows.append({"kode": kode, "label": label_map.get(kode, ""),
                     "value_labels": "; ".join(f"{k}={v}" for k, v in list(vl.items())[:10])})
    return pd.DataFrame(rows)

def cetak_ringkasan(sumber, df):
    def n(*kunci):
        return int(df["status"].apply(lambda s: any(k in s for k in kunci)).sum())
    print(f"\n  [{sumber.upper()}]  total entri kamus: {len(df)}")
    print(f"    Cocok / cocok-mirip ........ {n('COCOK')}")
    print(f"    Perlu tinjau (label beda) .. {n('PERLU TINJAU')}")
    print(f"    TIDAK DITEMUKAN ............. {n('TIDAK DITEMUKAN')}   <-- prioritas perbaikan")
    print(f"    Derivasi (komponen hilang) . {n('DERIVASI')} ({n('komponen HILANG')})")
    print(f"    Tidak tersedia (sesuai) .... {n('TIDAK TERSEDIA')}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print(f" VERIFIKASI KAMUS HARMONISASI ({len(K)} VARIABEL) vs METADATA .SAV")
    print("=" * 70)

    semua = []
    for sumber, path in FILES.items():
        if not os.path.exists(path):
            print(f"\n  ! Lewati {sumber}: file tidak ditemukan -> {path}")
            continue
        print(f"\n  Membaca metadata: {path} ...")
        meta = baca_metadata(path)
        print(f"    {meta.number_columns} variabel, {meta.number_rows} baris terdeteksi")

        df = verifikasi_sumber(sumber, meta)
        df.insert(0, "sumber", sumber)
        df.to_csv(os.path.join(OUTPUT_DIR, f"laporan_verifikasi_{sumber}.csv"),
                  index=False, encoding="utf-8-sig")
        semua.append(df)

        dump_metadata_lengkap(meta).to_csv(
            os.path.join(OUTPUT_DIR, f"metadata_lengkap_{sumber}.csv"),
            index=False, encoding="utf-8-sig")

        pii = deteksi_pii(meta)
        pii.to_csv(os.path.join(OUTPUT_DIR, f"kandidat_pii_{sumber}.csv"),
                   index=False, encoding="utf-8-sig")
        if len(pii):
            print(f"    ! {len(pii)} variabel terindikasi PII -> kandidat_pii_{sumber}.csv")

        cetak_ringkasan(sumber, df)

    if semua:
        gab = pd.concat(semua, ignore_index=True)
        gab.to_csv(os.path.join(OUTPUT_DIR, "laporan_verifikasi_gabungan.csv"),
                   index=False, encoding="utf-8-sig")

        def status_grup(s):
            if "TIDAK DITEMUKAN" in s: return "tidak_ditemukan"
            if "PERLU TINJAU" in s:    return "perlu_tinjau"
            if "COCOK" in s:           return "cocok"
            if "komponen HILANG" in s: return "derivasi_hilang"
            if "DERIVASI" in s:        return "derivasi_ok"
            return "tidak_tersedia"
        gab["status_grup"] = gab["status"].apply(status_grup)
        pivot = gab.pivot_table(index=["sumber", "kategori"], columns="status_grup",
                                values="nama_standar", aggfunc="count", fill_value=0)
        pivot.to_csv(os.path.join(OUTPUT_DIR, "ringkasan_per_kategori.csv"),
                     encoding="utf-8-sig")
        print(f"\n  Semua laporan tersimpan di folder: {OUTPUT_DIR}/")
    else:
        print("\n  Tidak ada file yang berhasil diproses. Cek path di KONFIGURASI.")
    print("\n  Selesai.\n")

if __name__ == "__main__":
    main()
