# app.py — Dashboard CSV robuste + Barres comparatives + Linéaire simplifié
import csv
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
        dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t")
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
            except:
                return chunk.decode("latin-1", errors="ignore")
        return str(chunk)
    else:
        with open(file_like, "rb") as f:
            chunk = f.read(4096)
        try:
            return chunk.decode("utf-8", errors="ignore")
        except:
            return chunk.decode("latin-1", errors="ignore")

def _clean_columns(df):
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

def _coerce_numeric_with_commas(df):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            s = out[c].astype(str).str.strip()
            if s.str.len().median() > 50: continue
            s2 = s.str.replace("\u00a0", " ").str.replace(" ", "").str.replace(",", ".")
            conv = pd.to_numeric(s2, errors="coerce")
            if conv.notna().mean() >= 0.7:
                out[c] = conv
    return out

@st.cache_data(show_spinner=False)
def load_csv_robust(file_like, manual_sep=None):
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
    for alt in [",", ";", "\t", "|"]:
        if alt == sep: continue
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
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= max_uniques]

# ============= INTERFACE =============

st.title("📊 CSV Dashboard robuste")
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
    st.warning("Le fichier est vide.")
    st.stop()

st.success("✅ Fichier chargé !")
st.write("Aperçu :", df.head(30))

num_cols = get_num_cols(df)
cat_cols = get_cat_cols(df)

k1, k2, k3 = st.columns(3)
k1.metric("Lignes", f"{len(df):,}".replace(",", " "))
k2.metric("Colonnes", str(df.shape[1]))
k3.metric("Numériques", str(len(num_cols)))

# ============= ONGLET 1 à 4 =============

tab1, tab2, tab3, tab4 = st.tabs(["📊 Barres", "🟢 Scatter", "🥧 Camembert", "📈 Linéaire"])

# ---------- Barres ----------
with tab1:
    st.subheader("Barres")
    mode = st.radio("Mode", ["Agrégation (1 mesure)", "Comparatif (2 mesures)"], horizontal=True)

    if mode == "Agrégation (1 mesure)":
        if not num_cols or not cat_cols:
            st.info("Besoin d'au moins 1 colonne numérique et 1 catégorielle.")
        else:
            g = st.selectbox("Catégorie (X)", cat_cols)
            y = st.selectbox("Mesure (Y)", num_cols)
            how = st.selectbox("Agrégation", ["sum", "mean", "median", "min", "max"])
            topn = st.slider("Top N", 3, 30, 10)
            df_bar = df.groupby(g)[y].agg(how).sort_values(ascending=False).head(topn)
            fig, ax = plt.subplots()
            ax.bar(df_bar.index.astype(str), df_bar.values)
            ax.set_xlabel(g); ax.set_ylabel(y); ax.set_title(f"{how} de {y} par {g}")
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig)

    else:
        if len(num_cols) < 2 or not cat_cols:
            st.info("2 colonnes numériques et 1 catégorielle sont nécessaires.")
        else:
            g = st.selectbox("Catégorie (X)", cat_cols)
            y1 = st.selectbox("Mesure 1", num_cols)
            y2 = st.selectbox("Mesure 2", [c for c in num_cols if c != y1])
            how = st.selectbox("Agrégation", ["sum", "mean", "median", "min", "max"])
            base = df.groupby(g)[[y1, y2]].agg(how)
            base["__tri__"] = base[[y1, y2]].max(axis=1)
            base = base.sort_values("__tri__", ascending=False).drop(columns="__tri__")
            topn = st.slider("Top N", 3, 30, 10)
            base = base.head(topn)
            idx = np.arange(len(base))
            width = 0.4
            fig, ax = plt.subplots()
            ax.bar(idx - width/2, base[y1], width, label=y1)
            ax.bar(idx + width/2, base[y2], width, label=y2)
            ax.set_xticks(idx, base.index.astype(str), rotation=30, ha="right")
            ax.legend()
            st.pyplot(fig)

# ---------- Scatter ----------
with tab2:
    st.subheader("Nuage de points")
    if len(num_cols) < 2:
        st.info("Deux colonnes numériques sont nécessaires.")
    else:
        x = st.selectbox("Axe X", num_cols)
        y = st.selectbox("Axe Y", [c for c in num_cols if c != x])
        fig, ax = plt.subplots()
        ax.scatter(df[x], df[y], alpha=0.7)
        ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"{y} vs {x}")
        st.pyplot(fig)

# ---------- Camembert ----------
with tab3:
    st.subheader("Répartition (camembert)")
    if not cat_cols:
        st.info("Au moins une colonne catégorielle est requise.")
    else:
        c = st.selectbox("Catégorie", cat_cols)
        series = df[c].astype(str).value_counts().head(12)
        fig, ax = plt.subplots()
        ax.pie(series, labels=series.index, autopct="%.1f%%", startangle=90)
        ax.set_title(f"Répartition de {c}")
        st.pyplot(fig)

# ---------- Linéaire (simplifié) ----------
with tab4:
    st.subheader("Courbes (linéaire)")
    xcol = st.selectbox("Axe X", df.columns)
    ycols = st.multiselect("Séries Y", num_cols, default=num_cols[:1])
    if not ycols:
        st.info("Choisis au moins une série Y.")
    else:
        d = df[[xcol] + ycols].dropna()
        try: d = d.sort_values(xcol)
        except: pass
        fig, ax = plt.subplots()
        for c in ycols:
            ax.plot(d[xcol].astype(str), d[c], label=c)
        ax.set_xlabel(xcol); ax.set_ylabel("Valeur")
        ax.set_title("Courbe(s) linéaire(s)")
        plt.xticks(rotation=30, ha="right")
        ax.legend()
        st.pyplot(fig)
