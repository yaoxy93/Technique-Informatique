# app.py — Dashboard CSV robuste (auto-détection du séparateur)
# -------------------------------------------------------------
# Points clés :
# - Auto-détection du séparateur via csv.Sniffer() + fallback intelligent
# - Gère UTF-8/UTF-8-SIG/Latin-1, guillemets, lignes problématiques
# - Nettoyage des colonnes + conversion numérique (virgules décimales -> points)
# - Sélecteur manuel de séparateur en secours (sidebar)
# - Mini dashboard (aperçu + 3 graphes de base) pour valider les données

import csv
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="CSV Auto-Sep Dashboard", page_icon="📊", layout="wide")

# ============= UTILS =============

def _read_try(file_like, **opts):
    """Essaye une lecture avec pandas et retourne (df, err). Reset le pointeur si nécessaire."""
    err = None
    # Important avec st.file_uploader: il faut remettre le curseur au début avant chaque tentative
    if hasattr(file_like, "seek"):
        file_like.seek(0)
    try:
        df = pd.read_csv(file_like, **opts)
        return df, None
    except Exception as e:
        err = e
    return pd.DataFrame(), err

def _detect_sep_from_sample(sample: str) -> str:
    """Devine le séparateur dans un échantillon texte; fallback ';' si doute."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t,")
        return dialect.delimiter
    except Exception:
        # Heuristique simple : si beaucoup de ';' -> ';', sinon ','
        counts = {sep: sample.count(sep) for sep in [",", ";", "\t", "|"]}
        # On préfère ';' en Europe quand les comptes sont proches
        if counts[";"] >= counts[","]:
            return ";"
        return max(counts, key=counts.get) or ";"

def _read_text(file_like) -> str:
    """Lit un petit sample du fichier (bytes) et renvoie du texte utf-8 propre pour Sniffer."""
    if hasattr(file_like, "read"):
        pos = file_like.tell()
        chunk = file_like.read(4096)
        # reset
        file_like.seek(pos)
        if isinstance(chunk, bytes):
            try:
                return chunk.decode("utf-8", errors="ignore")
            except Exception:
                return chunk.decode("latin-1", errors="ignore")
        return str(chunk)
    else:
        # Chemin sur disque
        with open(file_like, "rb") as f:
            chunk = f.read(4096)
        try:
            return chunk.decode("utf-8", errors="ignore")
        except Exception:
            return chunk.decode("latin-1", errors="ignore")

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie et déduplique proprement les noms de colonnes."""
    cols = []
    seen = {}
    for c in df.columns:
        cc = str(c).strip()
        cc = cc.replace("\n", " ").replace("\r", " ")
        cc = re.sub(r"\s+", " ", cc)
        # déduplication si colonnes dupliquées
        base = cc
        k = 1
        while cc in seen:
            k += 1
            cc = f"{base} ({k})"
        seen[cc] = True
        cols.append(cc)
    df.columns = cols
    return df

def _coerce_numeric_with_commas(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit automatiquement en numériques les colonnes object avec virgule décimale (ex: '12,5')."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            s = out[c].astype(str).str.strip()
            # ignore les grandes chaînes type GeoJSON/URL
            if s.str.len().median() > 50:
                continue
            s2 = (
                s.str.replace("\u00a0", " ", regex=False)   # NBSP
                 .str.replace(" ", "", regex=False)         # espaces milliers
                 .str.replace(",", ".", regex=False)        # virgule -> point
            )
            try:
                conv = pd.to_numeric(s2, errors="coerce")
                # On garde si au moins 70% deviennent des nombres
                if conv.notna().mean() >= 0.7:
                    out[c] = conv
            except Exception:
                pass
    return out

@st.cache_data(show_spinner=False)
def load_csv_robust(file_like, manual_sep: str | None = None) -> pd.DataFrame:
    """
    Lecture très robuste d'un CSV (Streamlit uploader ou chemin).
    - Auto-détecte le séparateur (ou utilise manual_sep si fourni)
    - Essaie plusieurs encodages et options (quotes, on_bad_lines)
    - Nettoie les colonnes et tente des conversions numériques
    """
    sample = _read_text(file_like)
    sep = manual_sep or _detect_sep_from_sample(sample)

    tries = [
        # Essai standard
        dict(sep=sep, engine="python", encoding="utf-8-sig", quotechar='"'),
        # Variante encodage
        dict(sep=sep, engine="python", encoding="utf-8", quotechar='"'),
        dict(sep=sep, engine="python", encoding="latin-1", quotechar='"'),
        # Si des lignes sont malformées, on préfère ne pas planter
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

    # Dernier recours : retente avec séparateurs alternatifs
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

# ============= UI =============

st.title("📊 CSV Auto-Sep Dashboard (zéro prise de tête)")

with st.sidebar:
    st.header("⚙️ Paramètres d’import")
    manual_toggle = st.toggle("Forcer un séparateur", value=False, help="À cocher si l'auto-détection n'est pas bonne.")
    manual_sep = None
    if manual_toggle:
        manual_sep = st.radio("Séparateur", options=[";", ",", "\\t", "|"], index=0, horizontal=True)
        if manual_sep == "\\t":
            manual_sep = "\t"
    st.caption("Astuce : les données européennes sont souvent en `;`.")

file = st.file_uploader("Charge un fichier CSV", type=["csv"])
if not file:
    st.info("➡️ Uploade un CSV pour commencer.")
    st.stop()

df = load_csv_robust(file, manual_sep=manual_sep)

if df.empty:
    st.warning("Le fichier a été lu, mais le tableau est vide (ou toutes les lignes ont été ignorées).")
    st.stop()

st.success("✅ CSV chargé !")
st.write("Aperçu :", df.head(50))

# ============= MINI DASHBOARD pour valider =============

num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns if c not in num_cols and df[c].nunique(dropna=True) <= 100]

col1, col2, col3 = st.columns(3)
col1.metric("Lignes", f"{len(df):,}".replace(",", " "))
col2.metric("Colonnes", f"{df.shape[1]}")
col3.metric("Numériques", f"{len(num_cols)}")

tabs = st.tabs(["📊 Barres", "🟢 Nuage de points", "🥧 Camembert"])

with tabs[0]:
    st.subheader("Barres agrégées")
    if not num_cols or not cat_cols:
        st.info("Besoin d’au moins 1 numérique + 1 catégorielle.")
    else:
        g = st.selectbox("Grouper par", cat_cols, key="bar_g")
        y = st.selectbox("Mesure", num_cols, key="bar_y")
        how = st.selectbox("Agrégation", ["sum", "mean", "median", "min", "max"], index=0, key="bar_how")
        topn = st.slider("Top N", 3, 30, 10, key="bar_topn")
        data = df.groupby(g, dropna=False)[y].agg(how).sort_values(ascending=False).head(topn)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(data.index.astype(str), data.values)
        ax.set_xlabel(g); ax.set_ylabel(y); ax.set_title(f"{how} de {y} par {g}")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)

with tabs[1]:
    st.subheader("Nuage de points")
    if len(num_cols) < 2:
        st.info("Besoin d’au moins 2 colonnes numériques.")
    else:
        x = st.selectbox("X", num_cols, index=0, key="sc_x")
        y = st.selectbox("Y", num_cols, index=1 if len(num_cols)>1 else 0, key="sc_y")
        fig, ax = plt.subplots(figsize=(7,4))
        ax.scatter(df[x], df[y], alpha=0.85)
        ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"{y} vs {x}")
        st.pyplot(fig)

with tabs[2]:
    st.subheader("Répartition (camembert)")
    if not cat_cols:
        st.info("Besoin d’une colonne catégorielle.")
    else:
        c = st.selectbox("Catégorie", cat_cols, key="pie_c")
        series = df[c].astype(str).value_counts().head(12)
        fig, ax = plt.subplots(figsize=(6,6))
        ax.pie(series.values, labels=series.index, autopct="%.1f%%", startangle=90)
        ax.set_title(f"Répartition de {c} (Top 12)")
        st.pyplot(fig)
