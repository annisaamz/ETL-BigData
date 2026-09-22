# %% [markdown]
# # Analisis Data Kemacetan Lalu Lintas — Kota Metropolia
#
# **Tujuan proyek:** menurunkan tingkat kemacetan sebesar **15%**, dengan fokus pada *peak hours*.

#
# Dataset: `urban_traffic_congestion_travel_time.csv` (2.800 baris data per jam,
# 1 Jan 2024 – 26 Apr 2024).

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

DATA_PATH = "urban_traffic_congestion_travel_time.csv"
df = pd.read_csv(DATA_PATH)
df.head()

# %% [markdown]
# ## 1. Persiapan Data
#
# Membuat fitur turunan yang dibutuhkan untuk analisis: `hour`, `day_of_week`,
# `is_weekend`, dan `is_peak_hour`.

# %%
df["date_time"] = pd.to_datetime(df["date_time"])
df["hour"] = df["date_time"].dt.hour
df["date"] = df["date_time"].dt.date
df["day_of_week"] = df["date_time"].dt.day_name()
df["is_weekend"] = df["date_time"].dt.dayofweek >= 5
df["day_type"] = np.where(df["is_weekend"], "Weekend", "Weekday")
df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18, 19])

congestion_order = ["Low", "Medium", "High"]
df["congestion_level"] = pd.Categorical(
    df["congestion_level"], categories=congestion_order, ordered=True
)

print(f"Jumlah baris          : {len(df):,}")
print(f"Rentang tanggal       : {df['date_time'].min()} s.d. {df['date_time'].max()}")
print(f"Missing values        : {df.isna().sum().sum()}")
print(f"Zona kota             : {sorted(df['city_zone'].unique())}")
print(f"Tipe jalan            : {sorted(df['road_type'].unique())}")
print(f"Kondisi cuaca         : {sorted(df['weather_condition'].unique())}")

# %% [markdown]
# ---
# ## 2. Volume Kendaraan Berdasarkan Jam
#
# **Pertanyaan: Bagaimana volume kendaraan berdasarkan jam? Jam berapa traffic
# paling tinggi?**

# %%
hourly_volume = (
    df.groupby("hour")["traffic_volume"]
    .agg(["mean", "median", "std", "count"])
    .round(1)
    .rename(columns={"mean": "rata_rata", "median": "median", "std": "std_dev", "count": "n_observasi"})
)
hourly_volume_sorted = hourly_volume.sort_values("rata_rata", ascending=False)

top5_hours = hourly_volume_sorted.head(5)
print("5 jam dengan rata-rata volume kendaraan tertinggi:")
print(top5_hours)

jam_puncak = hourly_volume_sorted.index[0]
print(f"\n>> Traffic paling tinggi terjadi pada pukul {jam_puncak:02d}.00, "
      f"dengan rata-rata {hourly_volume_sorted.iloc[0]['rata_rata']:.0f} kendaraan/jam.")

# %%
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(hourly_volume.index, hourly_volume["rata_rata"], color="#3b6ea5", zorder=3)
ax.axvspan(6.5, 9.5, color="orange", alpha=0.15, label="Jam sibuk pagi (07–09)")
ax.axvspan(15.5, 19.5, color="red", alpha=0.15, label="Jam sibuk sore (16–19)")
ax.set_xticks(range(0, 24))
ax.set_xlabel("Jam")
ax.set_ylabel("Rata-rata volume kendaraan")
ax.set_title("Rata-rata Volume Kendaraan per Jam")
ax.legend()
plt.tight_layout()
plt.savefig("chart_volume_per_jam.png")
plt.show()

# %% [markdown]
# ---
# ## 3. Volume Traffic: Weekday vs Weekend
#
# **Pertanyaan: Apakah traffic lebih tinggi pada weekday atau weekend?**

# %%
daytype_summary = (
    df.groupby("day_type")["traffic_volume"]
    .agg(["mean", "median", "std", "count"])
    .round(1)
)
print(daytype_summary)

weekday_vals = df.loc[~df["is_weekend"], "traffic_volume"]
weekend_vals = df.loc[df["is_weekend"], "traffic_volume"]
t_stat, p_val = stats.ttest_ind(weekday_vals, weekend_vals, equal_var=False)

lebih_tinggi = "weekday" if weekday_vals.mean() > weekend_vals.mean() else "weekend"
signifikan = "signifikan secara statistik" if p_val < 0.05 else "TIDAK signifikan secara statistik"
print(f"\nRata-rata weekday : {weekday_vals.mean():.1f}")
print(f"Rata-rata weekend : {weekend_vals.mean():.1f}")
print(f"Uji t (Welch)     : t = {t_stat:.3f}, p-value = {p_val:.4f} -> perbedaan {signifikan}")
print(f">> Volume traffic secara rata-rata lebih tinggi pada {lebih_tinggi}.")

# %%
fig, ax = plt.subplots(figsize=(6, 5))
df.boxplot(column="traffic_volume", by="day_type", ax=ax, grid=False)
ax.set_title("Distribusi Volume Kendaraan: Weekday vs Weekend")
ax.set_ylabel("Volume kendaraan")
ax.set_xlabel("")
plt.suptitle("")
plt.tight_layout()
plt.savefig("chart_weekday_weekend.png")
plt.show()

# %% [markdown]
# ---
# ## 4. Rata-rata Kecepatan Berdasarkan Jam
#
# **Pertanyaan: Bagaimana rata-rata speed berdasarkan jam?**

# %%
hourly_speed = df.groupby("hour")["average_speed_kmph"].mean().round(1)
print(hourly_speed.sort_values().head(5).rename("kecepatan_rata_rata (km/jam)"))

jam_lambat = hourly_speed.idxmin()
print(f"\n>> Kecepatan rata-rata terendah terjadi pada pukul {jam_lambat:02d}.00 "
      f"({hourly_speed.min():.1f} km/jam) — konsisten dengan jam padat pada bagian 2.")

# %%
fig, ax1 = plt.subplots(figsize=(11, 5))
ax1.bar(hourly_volume.index, hourly_volume["rata_rata"], color="#c7d9ec", zorder=2, label="Volume (batang)")
ax1.set_xlabel("Jam")
ax1.set_ylabel("Rata-rata volume kendaraan", color="#3b6ea5")
ax1.set_xticks(range(0, 24))

ax2 = ax1.twinx()
ax2.plot(hourly_speed.index, hourly_speed.values, color="#d9534f", marker="o", linewidth=2, label="Kecepatan (garis)")
ax2.set_ylabel("Rata-rata kecepatan (km/jam)", color="#d9534f")

ax1.set_title("Volume Kendaraan vs Kecepatan Rata-rata per Jam")
fig.tight_layout()
plt.savefig("chart_volume_vs_speed.png")
plt.show()

# %% [markdown]
# Terlihat pola berlawanan arah (*inverse relationship*) yang jelas antara volume
# kendaraan dan kecepatan rata-rata — mengonfirmasi bahwa jam dengan volume tinggi
# adalah jam dengan kecepatan terendah (indikasi kemacetan)

# %%
corr_vol_speed = df["traffic_volume"].corr(df["average_speed_kmph"])
print(f"Korelasi Pearson volume vs kecepatan (level data mentah): r = {corr_vol_speed:.3f}")

# %% [markdown]
# ---
# ## 5. Analisis Congestion
# ### 5a. Kapan Congestion Paling Tinggi?

# %%
congestion_by_hour = (
    pd.crosstab(df["hour"], df["congestion_level"], normalize="index") * 100
).round(1)
print(congestion_by_hour)

jam_congestion_tertinggi = congestion_by_hour["High"].idxmax()
pct_tertinggi = congestion_by_hour["High"].max()
print(f"\n>> Proporsi congestion 'High' tertinggi terjadi pada pukul "
      f"{jam_congestion_tertinggi:02d}.00 ({pct_tertinggi:.1f}% dari observasi jam tersebut).")

# %%
fig, ax = plt.subplots(figsize=(12, 5))
colors = {"Low": "#5cb85c", "Medium": "#f0ad4e", "High": "#d9534f"}
bottom = np.zeros(24)
for level in congestion_order:
    vals = congestion_by_hour[level].reindex(range(24)).fillna(0).values
    ax.bar(range(24), vals, bottom=bottom, label=level, color=colors[level])
    bottom += vals
ax.set_xticks(range(24))
ax.set_xlabel("Jam")
ax.set_ylabel("Proporsi (%)")
ax.set_title("Komposisi Tingkat Congestion per Jam")
ax.legend(title="Congestion Level")
plt.tight_layout()
plt.savefig("chart_congestion_per_jam.png")
plt.show()

# %% [markdown]
# ### 5b. Apakah `is_peak_hour` Berhubungan dengan Congestion?

# %%
ct_peak = pd.crosstab(df["is_peak_hour"], df["congestion_level"])
ct_peak.index = ct_peak.index.map({True: "Jam Sibuk", False: "Bukan Jam Sibuk"})
ct_peak_pct = (ct_peak.div(ct_peak.sum(axis=1), axis=0) * 100).round(1)
print("Jumlah observasi:")
print(ct_peak)
print("\nProporsi (%):")
print(ct_peak_pct)

chi2, p_peak, dof, _ = stats.chi2_contingency(ct_peak)
signif = "SIGNIFIKAN" if p_peak < 0.05 else "tidak signifikan"
print(f"\nUji Chi-square: chi2 = {chi2:.2f}, df = {dof}, p-value = {p_peak:.6f} -> hubungan {signif}")

# %% [markdown]
# ### 5c. Apakah Congestion Berbeda Berdasarkan `road_type`?

# %%
ct_road = pd.crosstab(df["road_type"], df["congestion_level"])
ct_road_pct = (ct_road.div(ct_road.sum(axis=1), axis=0) * 100).round(1)
print("Proporsi congestion per tipe jalan (%):")
print(ct_road_pct)

chi2_r, p_road, dof_r, _ = stats.chi2_contingency(ct_road)
signif_r = "SIGNIFIKAN" if p_road < 0.05 else "tidak signifikan"
print(f"\nUji Chi-square: chi2 = {chi2_r:.2f}, df = {dof_r}, p-value = {p_road:.6f} -> hubungan {signif_r}")

# %% [markdown]
# ### 5d. Apakah Weather Condition Berhubungan dengan Traffic?
#
# Diuji dari dua sisi: (i) volume kendaraan berdasarkan cuaca, dan (ii) tingkat
# congestion berdasarkan cuaca.

# %%
weather_volume = df.groupby("weather_condition")["traffic_volume"].agg(["mean", "std", "count"]).round(1)
print("Volume kendaraan berdasarkan cuaca:")
print(weather_volume)

groups = [df.loc[df["weather_condition"] == w, "traffic_volume"] for w in df["weather_condition"].unique()]
f_stat, p_anova = stats.f_oneway(*groups)
print(f"\nANOVA volume ~ cuaca: F = {f_stat:.3f}, p-value = {p_anova:.4f} "
      f"-> {'signifikan' if p_anova < 0.05 else 'tidak signifikan'}")

ct_weather = pd.crosstab(df["weather_condition"], df["congestion_level"])
ct_weather_pct = (ct_weather.div(ct_weather.sum(axis=1), axis=0) * 100).round(1)
print("\nProporsi congestion per kondisi cuaca (%):")
print(ct_weather_pct)

chi2_w, p_weather, dof_w, _ = stats.chi2_contingency(ct_weather)
print(f"Uji Chi-square cuaca vs congestion: chi2 = {chi2_w:.2f}, df = {dof_w}, "
      f"p-value = {p_weather:.4f} -> {'signifikan' if p_weather < 0.05 else 'tidak signifikan'}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
weather_volume["mean"].sort_values().plot(kind="barh", ax=axes[0], color="#3b6ea5")
axes[0].set_xlabel("Rata-rata volume kendaraan")
axes[0].set_title("Volume Kendaraan per Kondisi Cuaca")

ct_weather_pct[congestion_order].plot(kind="bar", stacked=True, ax=axes[1],
                                       color=[colors[c] for c in congestion_order])
axes[1].set_ylabel("Proporsi (%)")
axes[1].set_title("Komposisi Congestion per Kondisi Cuaca")
axes[1].legend(title="Congestion")
plt.tight_layout()
plt.savefig("chart_weather.png")
plt.show()

# %% [markdown]
# ---
# ## 6. Analisis Accident
# ### 6a. Apakah Traffic Volume Berbeda Ketika Terjadi Accident?

# %%
accident_summary = df.groupby("accident_reported")["traffic_volume"].agg(["mean", "std", "count"]).round(1)
print(accident_summary)

vol_no_acc = df.loc[df["accident_reported"] == "No", "traffic_volume"]
vol_acc = df.loc[df["accident_reported"] == "Yes", "traffic_volume"]
t_acc, p_acc = stats.ttest_ind(vol_acc, vol_no_acc, equal_var=False)
print(f"\nUji t (Welch): t = {t_acc:.3f}, p-value = {p_acc:.4f} "
      f"-> perbedaan {'signifikan' if p_acc < 0.05 else 'tidak signifikan'}")

lebih_besar = "lebih tinggi saat ada accident" if vol_acc.mean() > vol_no_acc.mean() else "lebih rendah saat ada accident"
print(f">> Rata-rata volume kendaraan {lebih_besar} "
      f"({vol_acc.mean():.0f} vs {vol_no_acc.mean():.0f}).")

# %% [markdown]
# ### 6b. Apakah Accident Lebih Sering Terjadi pada Jam Tertentu?

# %%
accident_by_hour = (
    df.groupby("hour")["accident_reported"]
    .apply(lambda s: (s == "Yes").mean() * 100)
    .round(1)
    .rename("persentase_accident (%)")
)
print(accident_by_hour.sort_values(ascending=False).head(6))

jam_paling_rawan = accident_by_hour.idxmax()
print(f"\n>> Persentase kejadian accident tertinggi pada pukul {jam_paling_rawan:02d}.00 "
      f"({accident_by_hour.max():.1f}% dari observasi jam tersebut).")

accident_count_by_hour = df.groupby("hour")["accident_reported"].apply(lambda s: (s == "Yes").sum())
chi2_acc_hour = stats.chisquare(accident_count_by_hour)
print(f"\nUji Chi-square goodness-of-fit (distribusi accident merata sepanjang 24 jam?): "
      f"chi2 = {chi2_acc_hour.statistic:.2f}, p-value = {chi2_acc_hour.pvalue:.4f} -> "
      f"{'TIDAK merata (ada pola jam tertentu)' if chi2_acc_hour.pvalue < 0.05 else 'relatif merata di semua jam'}")

# %%
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(accident_by_hour.index, accident_by_hour.values, color="#8b3a3a", zorder=3)
ax.axvspan(6.5, 9.5, color="orange", alpha=0.12, label="Jam sibuk pagi")
ax.axvspan(15.5, 19.5, color="red", alpha=0.12, label="Jam sibuk sore")
ax.set_xticks(range(0, 24))
ax.set_xlabel("Jam")
ax.set_ylabel("Persentase kejadian accident (%)")
ax.set_title("Persentase Kejadian Accident per Jam")
ax.legend()
plt.tight_layout()
plt.savefig("chart_accident_per_jam.png")
plt.show()

# %% [markdown]
# ---
# ## 7. Ringkasan Temuan & Rekomendasi
#
# Ringkasan berikut dirangkai otomatis dari hasil analisis di atas untuk mendukung
# strategi pencapaian target **penurunan kemacetan 15%** pada proyek Kota Metropolia.

# %%
print("=" * 70)
print("RINGKASAN TEMUAN UTAMA")
print("=" * 70)
print(f"1. Volume kendaraan tertinggi     : pukul {jam_puncak:02d}.00 "
      f"({hourly_volume_sorted.iloc[0]['rata_rata']:.0f} kendaraan/jam)")
print(f"2. Volume weekday vs weekend      : {lebih_tinggi} lebih tinggi "
      f"(p = {p_val:.4f}, {signifikan})")
print(f"3. Kecepatan terendah             : pukul {jam_lambat:02d}.00 "
      f"({hourly_speed.min():.1f} km/jam)")
print(f"4. Congestion 'High' tertinggi    : pukul {jam_congestion_tertinggi:02d}.00 "
      f"({pct_tertinggi:.1f}%)")
print(f"5. is_peak_hour vs congestion     : hubungan {signif} (p = {p_peak:.6f})")
print(f"6. road_type vs congestion        : hubungan {signif_r} (p = {p_road:.6f})")
print(f"7. weather vs volume              : {'signifikan' if p_anova < 0.05 else 'tidak signifikan'} "
      f"(ANOVA p = {p_anova:.4f})")
print(f"8. weather vs congestion          : {'signifikan' if p_weather < 0.05 else 'tidak signifikan'} "
      f"(Chi-square p = {p_weather:.4f})")
print(f"9. Volume saat accident           : {lebih_besar} (p = {p_acc:.4f})")
print(f"10. Jam paling rawan accident     : pukul {jam_paling_rawan:02d}.00 "
      f"({accident_by_hour.max():.1f}%)")
print("=" * 70)

# %% [markdown]
# ### Implikasi untuk Target Penurunan Kemacetan 15%
#
# - **Fokuskan intervensi pada jam sibuk** yang teridentifikasi di atas (bukan
#   asumsi umum 07–09 & 16–19), karena data menunjukkan pola aktual volume,
#   kecepatan, dan congestion yang konsisten saling menguatkan.
# - Jika `road_type` terbukti berhubungan signifikan dengan congestion, prioritaskan
#   manajemen arus (rekayasa lalu lintas, ATCS, rambu dinamis) pada tipe jalan
#   dengan proporsi congestion "High" tertinggi.
# - Jika cuaca terbukti berhubungan signifikan dengan volume/congestion,
#   pertimbangkan sistem peringatan dini cuaca terintegrasi dengan rekomendasi
#   rute alternatif.
# - Jam dengan persentase accident tertinggi sebaiknya menjadi prioritas
#   penempatan personel lalu lintas/kepolisian dan kamera pengawas, karena
#   accident turut berkontribusi pada penurunan kecepatan dan peningkatan
#   congestion.
# - Hasil ini menjadi input kuantitatif untuk tahap *modelling* berikutnya
#   (Gradient Boosting → LSTM → ASTGCN) guna memprediksi titik dan waktu
#   potensi kemacetan secara lebih presisi.
