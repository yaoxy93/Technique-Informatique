import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk

st.set_page_config(page_title="Dashboard CSV simple", page_icon="📊", layout="wide")
st.title("📊 Dashboard CSV (aucune installation locale requise)")

file = st.file_uploader("Charge un fichier CSV", type=["csv"])
if not file:
    st.info("➡️ Uploade un CSV pour commencer.")
    st.stop()

@st.cache_data
def load_df(f):
    try:
        # Pandas devine bien ; , \t
        return pd.read_csv(f, engine="python")
    except Exception as e:
        st.error(f"Erreur de lecture du CSV : {e}")
        return pd.DataFrame()

df = load_df(file)
if df.empty:
    st.warning("Le fichier a été lu mais le tableau est vide.")
    st.stop()

st.success("✅ CSV chargé !")
st.dataframe(df.head(50), use_container_width=True)

# Détecte types
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns if c not in num_cols and df[c].nunique(dropna=True) <= 100]

st.markdown("### 🔎 Filtres rapides")
with st.expander("Ouvrir / masquer"):
    chosen_cats = {}
    for c in cat_cols[:2]:
        vals = sorted(df[c].dropna().astype(str).unique().tolist())
        chosen = st.multiselect(f"{c}", vals)
        if chosen:
            df = df[df[c].astype(str).isin(chosen)]
    if num_cols:
        n = st.selectbox("Filtrer une colonne numérique :", ["(aucun)"] + num_cols)
        if n != "(aucun)":
            lo, hi = float(df[n].min()), float(df[n].max())
            r = st.slider(f"Intervalle {n}", lo, hi, (lo, hi))
            df = df[df[n].between(*r)]

k1, k2 = st.columns(2)
k1.metric("Lignes", f"{len(df):,}".replace(",", " "))
k2.metric("Colonnes", f"{df.shape[1]}")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Barres", "🟢 Scatter", "🥧 Répartition", "🗺️ Carte"])

with tab1:
    st.subheader("Barres agrégées")
    if not num_cols or not cat_cols:
        st.info("Besoin d’au moins 1 numérique + 1 catégorielle.")
    else:
        g = st.selectbox("Grouper par", cat_cols)
        y = st.selectbox("Mesure", num_cols)
        how = st.selectbox("Agrégation", ["sum","mean","median","min","max"], index=0)
        topn = st.slider("Top N", 3, 30, 10)
        data = df.groupby(g, dropna=False)[y].agg(how).sort_values(ascending=False).head(topn)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(data.index.astype(str), data.values)
        ax.set_xlabel(g); ax.set_ylabel(y); ax.set_title(f"{how} de {y} par {g}")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)

with tab2:
    st.subheader("Nuage de points")
    if len(num_cols) < 2:
        st.info("Besoin d’au moins 2 colonnes numériques.")
    else:
        x = st.selectbox("X", num_cols, index=0)
        y = st.selectbox("Y", num_cols, index=1 if len(num_cols)>1 else 0)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.scatter(df[x], df[y], alpha=0.85)
        ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"{y} vs {x}")
        st.pyplot(fig)

with tab3:
    st.subheader("Répartition (camembert)")
    if not cat_cols:
        st.info("Besoin d’une colonne catégorielle.")
    else:
        c = st.selectbox("Catégorie", cat_cols)
        series = df[c].astype(str).value_counts().head(12)
        fig, ax = plt.subplots(figsize=(6,6))
        ax.pie(series.values, labels=series.index, autopct="%.1f%%", startangle=90)
        ax.set_title(f"Répartition de {c} (Top 12)")
        st.pyplot(fig)

with tab4:
    st.subheader("Carte (si lat/lon existent)")
    lat, lon = None, None
    for la, lo in [("lat","lon"),("latitude","longitude"),("Lat","Lon"),("Latitude","Longitude")]:
        if la in df.columns and lo in df.columns:
            lat, lon = la, lo
            break
    if lat and lon:
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(latitude=float(df[lat].mean()), longitude=float(df[lon].mean()), zoom=9),
            layers=[pdk.Layer("ScatterplotLayer", data=df, get_position=[lon, lat], get_radius=200, pickable=True)]
        ))
    else:
        st.info("Colonnes lat/lon non trouvées.")
