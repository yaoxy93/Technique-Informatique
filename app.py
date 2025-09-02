# app.py — Dashboard CSV robuste + Barres comparatives + Linéaire
# ---------------------------------------------------------------
# - Auto-détection du séparateur ; , | \t + encodage + quotes
# - Onglets : Barres (agrégation OU comparatif 2 mesures), Scatter, Camembert, Linéaire
# - Barres comparatives : 2 colonnes numériques face à face par catégorie
# - Linéaire : x = date/numérique/texte, multi-séries, resampling si datetime, lissage (moyenne glissante)

import csv
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="CSV Dashboard (auto-séparateur)", page_icon="📊", layout="wide")

# ============= UTILS =============

def _read_try(file_like, **opts):
    if hasattr(file_like, "seek"):
        file_like.seek(0)
    try:
        df = pd.read_csv(file_like, **opts)
        return df, None
    except Exception as e:
        return pd.DataFrame(), e

def _detect_sep_from_sample(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t,")
        return dialect.delimiter
    except Exception:
        counts = {sep: sample.count(sep) for sep in [",", ";", "\t", "|"]}
        if counts[";"] >= counts[","]:
            return ";"
        return max(counts, key=counts.get) or ";"

def _read_text(file_like) -> str:
    if hasattr(file_like, "read"):
        pos = file_like.tell()
        chunk = file_like.read(4096)
        file_like.seek(pos)
        if isinstance(chunk, bytes):
            try:
                return chunk.decode("utf-8", errors="ignore")
            except Exception:
                return chunk.decode("latin-1", errors="ignore")
        return str(chunk)
    else:
        with open(file_like, "rb") as f:
            chunk = f.read(4096)
        try:
            return chunk.decode("utf-8", errors="ignore")
        except Exception:
            return chunk.decode("latin-1", errors="ignore")

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols, seen = [], {}
    for c in df.columns:
        cc = str(c).strip().replace("\n", " ").replace("\r", " ")
        cc = re.sub(r"\s+", " ", cc)
        base, k = cc, 1
        while cc in seen:
            k += 1
            cc = f"{base} ({k})"
        seen[cc] = True
        cols.append(cc)
    df.columns = cols
    return df

def _coerce_numeric_with_commas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            s = out[c].astype(str).str.strip()
            if s.str.len().median() > 50:  # évite GeoJSON/URL longs
                continue
            s2 = (s.str.replace("\u00a0", " ", regex=False)
                    .str.replace(" ", "", regex=False)
                    .str.replace(",", ".", regex=False))
            conv = pd.to_numeric(s2, errors="coerce")
            if conv.notna().mean() >= 0.7:
                out[c] = conv
    return out

@st.cache_data(show_spinner=False)
def load_csv_robust(file_like, manual_sep: str | None = None) -> pd.DataFrame:
    sample = _read_text(file_like)
    sep = manual_sep or _detect_sep_from_sample(sample)

    tries = [
        dict(sep=sep, engine="python", encoding="utf-8-sig", quotechar='"'),
        dict(sep=sep, engine="python", encoding="utf-8", quotechar='"'),
        dict(sep=sep, engine="python", encoding="latin-1", quotechar='"'),
        dict(sep=sep, engine="python", encoding="utf-8-sig", quotechar='"', on_bad_lines="skip"),
    ]

    last_err = None
    for opts in tries:
        df, err = _read_try(file_like, **opts)
        if err is None and not df.empty:
            df = _clean_columns(df)
            df = _coerce_numeric_with_commas(df)
            return df
        last_err = err

    # retente avec séparateurs alternatifs
    for alt in [",", ";", "\t", "|"]:
        if alt == sep:
            continue
        df, err = _read_try(file_like, sep=alt, engine="python", encoding="utf-8-sig", quotechar='"')
        if err is None and not df.empty:
            df = _clean_columns(df)
            df = _coerce_numeric_with_commas(df)
            return df
        last_err = err

    st.error(f"Erreur de lecture du CSV : {last_err}")
    return pd.DataFrame()

def get_num_cols(df): 
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def get_cat_cols(df, max_uniques=100):
    cats=[]
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) <= max_uniques:
            cats.append(c)
    return cats

# ============= UI IMPORT =============

st.title("📊 CSV Dashboard")
with st.sidebar:
    st.header("⚙️ Import")
    manual_toggle = st.toggle("Forcer un séparateur", value=False)
    manual_sep = None
    if manual_toggle:
        manual_sep = st.radio("Séparateur", options=[";", ",", "\\t", "|"], index=0, horizontal=True)
        if manual_sep == "\\t":
            manual_sep = "\t"
    st.caption("Astuce : beaucoup de CSV européens utilisent `;`.")

file = st.file_uploader("Charge un fichier CSV", type=["csv"])
if not file:
    st.info("➡️ Uploade un CSV pour commencer.")
    st.stop()

df = load_csv_robust(file, manual_sep=manual_sep)
if df.empty:
    st.warning("Le fichier a été lu, mais le tableau est vide.")
    st.stop()

st.success("✅ CSV chargé !")
st.write("Aperçu :", df.head(30))

num_cols = get_num_cols(df)
cat_cols = get_cat_cols(df)

k1, k2, k3 = st.columns(3)
k1.metric("Lignes", f"{len(df):,}".replace(",", " "))
k2.metric("Colonnes", f"{df.shape[1]}")
k3.metric("Numériques", f"{len(num_cols)}")

# ============= ONGLET 1 : BARRES (agrégation ou comparatif) =============
tab1, tab2, tab3, tab4 = st.tabs(["📊 Barres", "🟢 Nuage de points", "🥧 Camembert", "📈 Linéaire"])

with tab1:
    st.subheader("Barres")
    mode = st.radio("Mode", ["Agrégation (1 mesure)", "Comparatif (2 mesures)"], horizontal=True)

    if mode == "Agrégation (1 mesure)":
        if not num_cols or not cat_cols:
            st.info("Besoin d’au moins 1 numérique + 1 catégorielle.")
        else:
            g = st.selectbox("Grouper par (catégoriel)", cat_cols, key="bar_g")
            y = st.selectbox("Mesure (numérique)", num_cols, key="bar_y")
            how = st.selectbox("Agrégation", ["sum", "mean", "median", "min", "max"], index=0, key="bar_how")
            topn = st.slider("Top N (après agrégation)", 3, 30, 10, key="bar_topn")
            df_bar = df.groupby(g, dropna=False)[y].agg(how).sort_values(ascending=False).head(topn)
            fig, ax = plt.subplots(figsize=(8,4))
            ax.bar(df_bar.index.astype(str), df_bar.values)
            ax.set_xlabel(g); ax.set_ylabel(y); ax.set_title(f"{how} de {y} par {g}")
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig)

    else:  # Comparatif (2 mesures)
        if len(num_cols) < 2 or not cat_cols:
            st.info("Besoin d’au moins 1 catégorielle + 2 numériques.")
        else:
            g = st.selectbox("Catégorie (X)", cat_cols, key="cmp_g")
            y1 = st.selectbox("Mesure 1", num_cols, key="cmp_y1")
            y2 = st.selectbox("Mesure 2", [c for c in num_cols if c != y1], key="cmp_y2")
            how = st.selectbox("Agrégation", ["sum", "mean", "median", "min", "max"], index=0, key="cmp_how")
            base = df.groupby(g, dropna=False)[[y1, y2]].agg(how)
            # tri sur la plus grande des deux (pour lisibilité)
            base["__tri__"] = base[[y1, y2]].max(axis=1)
            base = base.sort_values("__tri__", ascending=False).drop(columns="__tri__")
            topn = st.slider("Top N (catégories)", 3, min(30, len(base)), min(10, len(base)))
            base = base.head(topn)

            # barres côte à côte
            idx = np.arange(len(base))
            width = 0.42
            fig, ax = plt.subplots(figsize=(10,5))
            ax.bar(idx - width/2, base[y1].values, width, label=y1)
            ax.bar(idx + width/2, base[y2].values, width, label=y2)
            ax.set_xticks(idx, base.index.astype(str), rotation=30, ha="right")
            ax.set_xlabel(g); ax.set_ylabel("Valeur"); ax.set_title(f"{how} de {y1} & {y2} par {g}")
            ax.legend()
            st.pyplot(fig)

            

# ============= ONGLET 2 : Nuage de points =============
with tab2:
    st.subheader("Nuage de points (X et Y numériques)")
    if len(num_cols) < 2:
        st.info("Besoin d’au moins 2 colonnes numériques.")
    else:
        x = st.selectbox("Axe X (num.)", num_cols, key="sc_x")
        y = st.selectbox("Axe Y (num.)", [c for c in num_cols if c != x], key="sc_y")
        fig, ax = plt.subplots(figsize=(7,4))
        ax.scatter(df[x], df[y], alpha=0.85)
        ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"{y} vs {x}")
        st.pyplot(fig)

# ============= ONGLET 3 : CAMEMBERT =============
with tab3:
    st.subheader("Répartition (camembert)")
    if not cat_cols:
        st.info("Besoin d’au moins 1 catégorielle.")
    else:
        c = st.selectbox("Catégorie", cat_cols, key="pie_c")
        topn = st.slider("Top catégories à afficher", 3, 20, 12)
        series = df[c].astype(str).value_counts().head(topn)
        fig, ax = plt.subplots(figsize=(6,6))
        ax.pie(series.values, labels=series.index, autopct="%.1f%%", startangle=90)
        ax.set_title(f"Répartition de {c} — {mesure_mode if mesure_mode!='Comptage (count)' else 'count'}")
        st.pyplot(fig)

# ============= ONGLET 4 : LINEAIRE =============
with tab4:
    st.subheader("Courbes (linéaire)")
    # choix X (peut être date/numérique/texte) et Y (une ou plusieurs colonnes numériques)
    xcol = st.selectbox("Axe X", df.columns, key="ln_x")
    ycols = st.multiselect("Séries (Y)", num_cols, default=num_cols[:1], key="ln_y")
    if not ycols:
        st.info("Choisis au moins une série Y.")
    else:
        # tentative de conversion datetime si possible
        x = df[xcol]
        x_dt = pd.to_datetime(x, errors="coerce", dayfirst=True)
        use_dt = x_dt.notna().mean() > 0.7  # si >=70% parse OK, on considère datetime

        if use_dt:
            # options de resampling
            freq = st.selectbox("Regroupement temporel (resample)", ["Aucun", "Jour", "Semaine", "Mois", "Trimestre", "Année"], index=0)
            d = df.copy()
            d["_x"] = x_dt
            d = d.dropna(subset=["_x"])
            d = d.sort_values("_x").set_index("_x")
            if freq != "Aucun":
                fmap = {"Jour":"D","Semaine":"W","Mois":"M","Trimestre":"Q","Année":"Y"}
                d = d[ycols].resample(fmap[freq]).mean()
                to_plot = d
            else:
                to_plot = d[ycols]

            for c in ycols:
                ax.plot(to_plot.index, to_plot[c], label=c)
            ax.set_xlabel("Temps"); ax.set_ylabel("Valeur"); ax.set_title("Courbe(s) temporelle(s)")
            ax.legend()
            st.pyplot(fig)
        else:
            # X non temporel : on trie par X pour une courbe lisible
            d = df[[xcol] + ycols].dropna()
            try:
                d = d.sort_values(xcol)
            except Exception:
                pass
            fig, ax = plt.subplots(figsize=(9,4))
            for c in ycols:
                ax.plot(d[xcol].astype(str), d[c], label=c)
            ax.set_xlabel(xcol); ax.set_ylabel("Valeur"); ax.set_title("Courbe(s)")
            plt.xticks(rotation=30, ha="right")
            ax.legend()
            st.pyplot(fig)
