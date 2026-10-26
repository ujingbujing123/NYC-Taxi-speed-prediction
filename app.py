"""NYC Taxi: estimasi kecepatan perjalanan & peta kemacetan berdasarkan jam, hari, dan zona."""
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_ID = dict(zip(DAYS, ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]))
RUSH = [7, 8, 9, 16, 17, 18, 19]
LEVEL_COLOR = {"Macet Parah": "#c0392b", "Padat": "#e67e22", "Lancar": "#27ae60", "Sangat Lancar": "#2980b9"}

st.set_page_config(page_title="NYC Taxi: Kecepatan & Kemacetan", page_icon="🚕", layout="wide")


@st.cache_resource
def load_artifacts():
    names = ["speed_model.joblib", "strict_model.joblib", "meta.joblib", "app_data.joblib"]
    missing = [n for n in names if not (BASE / n).exists()]
    if missing:
        return None, missing
    return [joblib.load(BASE / n) for n in names], []


arts, missing = load_artifacts()
if arts is None:
    st.error(f"File artefak belum ada di folder aplikasi: {', '.join(missing)}. Jalankan sel ekspor di notebook, lalu salin file-nya ke sini.")
    st.stop()
model_a, model_s, meta, app = arts
agg = app["agg"]


def zname(z):
    return meta["zone_name"].get(int(z), f"Zona {int(z)}")


def level(v, thr):
    q1, q2, q3 = thr
    return "Macet Parah" if v < q1 else "Padat" if v < q2 else "Lancar" if v < q3 else "Sangat Lancar"


def time_feats(hour, day):
    return {"hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
            "is_rush": int(hour in RUSH), "is_weekend": int(day in ("Saturday", "Sunday"))}


def predict_route(pu, do, hour, day, passengers, distance=None):
    """Model A: estimasi kecepatan perjalanan. Biaya zona didekati dari median historis zona."""
    fp, fd = app["fee_pu"].get(pu, {}), app["fee_do"].get(do, {})
    row = {
        "passenger_count": passengers, **time_feats(hour, day),
        "same_zone": int(pu == do), "is_airport": int(pu in meta["airport_zones"] or do in meta["airport_zones"]),
        "congestion_surcharge": max(fp.get("congestion_surcharge", 0.0), fd.get("congestion_surcharge", 0.0)),
        "cbd_congestion_fee": max(fp.get("cbd_congestion_fee", 0.0), fd.get("cbd_congestion_fee", 0.0)),
        "Airport_fee": fp.get("Airport_fee", 0.0),
        "PULocationID": pu, "DOLocationID": do, "day_of_week": day,
        "RatecodeID": 2 if 132 in (pu, do) else 3 if 1 in (pu, do) else 1,
    }
    if meta["use_distance"]:
        row["trip_distance"] = distance or 2.0
    return float(model_a.predict(pd.DataFrame([row])[meta["features"]])[0])


def predict_zones(hour, day):
    """Model strict: hanya jam, hari, dan zona pickup."""
    zones = sorted(agg["PULocationID"].unique())
    d = pd.DataFrame({"PULocationID": zones, "day_of_week": day, **time_feats(hour, day)})
    d["pred"] = model_s.predict(d[["hour_sin", "hour_cos", "is_rush", "PULocationID", "day_of_week", "is_weekend"]])
    return d


st.title("🚕 NYC Taxi: Kecepatan & Tingkat Kemacetan")
st.caption(f"Data {app['n_rows']:,} perjalanan, {app['period'][0]} s/d {app['period'][1]}. "
           "Kecepatan taksi adalah proksi kemacetan, bukan pengukuran lalu lintas langsung, dan hasilnya adalah estimasi historis.")

tab_map, tab_trip, tab_insight = st.tabs(["🗺️ Peta kemacetan zona", "🧭 Estimasi perjalanan", "📊 Insight & model"])

# ---------------------------------------------------------------- TAB 1
with tab_map:
    st.subheader("Zona mana yang paling lambat pada jam dan hari tertentu?")
    c1, c2, c3 = st.columns([2, 3, 2])
    day = c1.selectbox("Hari", DAYS, index=4, format_func=DAY_ID.get, key="map_day")
    hour = c2.slider("Jam pickup", 0, 23, 17, key="map_hour")
    topn = c3.slider("Tampilkan N zona terlambat", 5, 30, 15)

    d = predict_zones(hour, day)
    obs = agg[(agg["day_of_week"] == day) & (agg["pickup_hour"] == hour)][["PULocationID", "median_speed", "trips"]]
    d = d.merge(obs, on="PULocationID", how="left").fillna({"trips": 0})
    d = d[d["trips"] >= app["min_cell"]].copy()
    d["Level"] = d["pred"].map(lambda v: level(v, app["map_thr"]))
    d["Indeks vs normal zona"] = (d["pred"] / d["PULocationID"].map(app["zone_overall"])).round(2)
    d["Zona"] = d["PULocationID"].map(zname)
    show = d.sort_values("pred").head(topn)

    m1, m2, m3 = st.columns(3)
    m1.metric("Zona dianalisis", f"{len(d)}")
    m2.metric("Zona berlevel Macet Parah", f"{(d['Level'] == 'Macet Parah').sum()}")
    m3.metric("Median prediksi kota", f"{d['pred'].median():.1f} mph")

    chart = (alt.Chart(show).mark_bar().encode(
        x=alt.X("pred:Q", title="Prediksi kecepatan (mph)"),
        y=alt.Y("Zona:N", sort="x", title=None),
        color=alt.Color("Level:N", scale=alt.Scale(domain=list(LEVEL_COLOR), range=list(LEVEL_COLOR.values()))),
        tooltip=["Zona", alt.Tooltip("pred:Q", format=".1f", title="Prediksi"), alt.Tooltip("median_speed:Q", title="Median historis"),
                 "Level", "Indeks vs normal zona", "trips"]).properties(height=28 * len(show) + 40))
    st.altair_chart(chart, width="stretch")
    st.dataframe(show[["Zona", "pred", "median_speed", "Indeks vs normal zona", "Level", "trips"]].rename(
        columns={"pred": "Prediksi (mph)", "median_speed": "Median historis (mph)", "trips": "Jumlah trip"}).round(2),
        hide_index=True, width="stretch")
    st.caption("Level memakai kuantil 25/50/75 dari prediksi seluruh kombinasi zona, hari, dan jam dalam seminggu (berbobot jumlah trip), sehingga pada jam tersibuk wajar bila banyak zona berlevel Macet Parah. "
               "Indeks di bawah 1 berarti lebih lambat dari kecepatan normal zona itu.")

    st.markdown("#### Pola mingguan satu zona")
    zlist = sorted(agg["PULocationID"].unique())
    default = int(d.sort_values("pred")["PULocationID"].iloc[0]) if len(d) else zlist[0]
    z = st.selectbox("Zona", zlist, index=zlist.index(default), format_func=zname)
    hz = agg[agg["PULocationID"] == z].copy()
    hz["Hari"] = hz["day_of_week"].map(DAY_ID)
    heat = (alt.Chart(hz).mark_rect().encode(
        x=alt.X("pickup_hour:O", title="Jam"), y=alt.Y("Hari:N", sort=list(DAY_ID.values()), title=None),
        color=alt.Color("median_speed:Q", scale=alt.Scale(scheme="redyellowgreen"), title="mph"),
        tooltip=["Hari", "pickup_hour", "median_speed", "trips"]).properties(height=230))
    st.altair_chart(heat, width="stretch")

# ---------------------------------------------------------------- TAB 2
with tab_trip:
    st.subheader("Estimasi kecepatan & durasi perjalanan")
    ids = sorted(meta["zone_name"]) if meta["zone_name"] else list(range(1, 264))
    c1, c2 = st.columns(2)
    pu = c1.selectbox("Zona asal (pickup)", ids, index=ids.index(161) if 161 in ids else 0, format_func=zname)
    do = c2.selectbox("Zona tujuan (dropoff)", ids, index=ids.index(237) if 237 in ids else 0, format_func=zname)
    c3, c4, c5 = st.columns(3)
    day2 = c3.selectbox("Hari", DAYS, index=4, format_func=DAY_ID.get, key="trip_day")
    hour2 = c4.slider("Jam berangkat", 0, 23, 18, key="trip_hour")
    pax = c5.number_input("Jumlah penumpang", 1, 6, 1)

    pair = app["pair_dist"].get((pu, do))
    dist = pair if pair else meta["median_dist_by_zone"].get(pu, 2.0)
    if meta["use_distance"]:
        dist = st.number_input("Jarak (mil)", 0.1, 50.0, float(dist), 0.1)

    v = predict_route(pu, do, hour2, day2, pax, dist)
    lv = level(v, meta["thresholds"])
    r1, r2, r3 = st.columns(3)
    r1.metric("Prediksi kecepatan", f"{v:.1f} mph")
    r2.metric("Tingkat kemacetan", lv)
    r3.metric("Estimasi durasi", f"{dist / v * 60:.0f} menit")
    st.caption(("Jarak adalah median historis pasangan zona ini." if pair else
                "Pasangan zona ini jarang muncul di data; jarak dipakai dari median zona asal, jadi durasi kurang akurat.")
               + f" Jarak dipakai: {dist:.1f} mil. Level mengikuti ambang kuartil per-perjalanan: "
               f"< {meta['thresholds'][0]} | < {meta['thresholds'][1]} | < {meta['thresholds'][2]} mph.")
    st.info(f"Model rute memprediksi terlalu cepat sekitar {app['bias_A']:+.2f} mph pada periode uji (Okt–Nov), "
            "karena kecepatan di akhir tahun lebih rendah daripada rata-rata data latih.")

# ---------------------------------------------------------------- TAB 3
with tab_insight:
    st.subheader("Pola kecepatan")
    a, b = st.columns(2)
    a.altair_chart(alt.Chart(app["hour_tbl"]).mark_line(point=True).encode(
        x=alt.X("pickup_hour:O", title="Jam"), y=alt.Y("median:Q", title="Median kecepatan (mph)", scale=alt.Scale(zero=False)),
        tooltip=["pickup_hour", "median", "mean"]).properties(title="Per jam", height=260), width="stretch")
    dd = app["day_tbl"].assign(Hari=lambda x: x["day_of_week"].map(DAY_ID))
    b.altair_chart(alt.Chart(dd).mark_bar().encode(
        x=alt.X("Hari:N", sort=list(DAY_ID.values()), title=None), y=alt.Y("speed_mph:Q", title="Median (mph)"),
        tooltip=["Hari"]).properties(title="Per hari", height=260), width="stretch")
    mt = app["month_tbl"]
    st.altair_chart(alt.Chart(mt).mark_line(point=True).encode(
        x=alt.X(f"{mt.columns[0]}:N", title="Bulan"), y=alt.Y(f"{mt.columns[1]}:Q", title="Median (mph)", scale=alt.Scale(zero=False))
    ).properties(title="Per bulan (kecepatan menurun di akhir tahun)", height=240), width="stretch")

    st.subheader("Evaluasi model (data uji berbasis waktu)")
    st.dataframe(app["results"], hide_index=True, width="stretch")
    s = app["strict_score"]
    st.markdown(f"**Model peta zona (hanya jam, hari, zona pickup):** MAE {s['MAE']:.2f} | RMSE {s['RMSE']:.2f} | R² {s['R2']:.3f}")
    st.markdown("**Ablasi `trip_distance`:**")
    st.dataframe(app["ablation"], hide_index=True, width="stretch")
    st.markdown(
        "**Keterbatasan:** (1) kecepatan taksi hanyalah proksi kemacetan; perjalanan pendek lambat karena lampu merah, bukan selalu macet. "
        "(2) Model rute memakai zona tujuan dan biaya zona yang secara tidak langsung memuat informasi jarak. "
        "(3) Data historis Jan–Nov 2025, bukan aliran real-time. (4) Biaya per zona didekati dari median historis."
    )
