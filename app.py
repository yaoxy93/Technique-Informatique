# app.py — Dashboard CSV robuste + Barres comparatives + Linéaire simple
# ----------------------------------------------------------------------
# - Lecture robuste : auto-détection du séparateur ; , | \t + encodage + quotes (+ fallback)
# - 4 onglets : Barres (agrégation OU comparatif 2 mesures), Nuage de point, Camembert, Linéaire
# - "Top catégories/points" ajouté partout pour garder des graphiques lisibles
# - Linéaire (simplifié) : X = n’importe quelle colonne, Y = 1..n colonnes numériques (pas de resampling/lissage)

import csv
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="CSV Dashboard (auto-séparateur)", page_icon="📊", layout="wide")

# =========================
#         UTILS
# =========================

def _read_try(file_like, **opts):
    """Tente une lecture pandas et renvoie (df, err). Remet le pointeur au début si nécessaire."""
    if hasattr(file_like, "seek"):
        file_like.seek(0)
    try:
        df = pd.read_csv(file_like, **opts)
        return df, None
    except Exception as e:
        return pd.DataFrame(), e

def _detect_sep_from_sample(sample: str) -> str:
    """Devine le séparateur à partir d’un petit échantillon (Sniffer + comptage simple)."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t,")
        return dialect.delimiter
    except Exception:
        counts = {sep: sample.count(sep) for sep in [",", ";", "\t", "|"]}
        # En Europe, si match serré, on préfère le ';'
        if counts[";"] >= counts[","]:
            return ";"
        return max(counts, key=counts.get) or ";"

def _read_text(file_like) -> str:
    """Lit ~4KB pour détecter séparateur/encodage, renvoie du texte UTF-8/Latin-1."""
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
    """Nettoie et déduplique les noms de colonnes (espaces multiples, retours-ligne, doublons)."""
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
    """Convertit en numérique les colonnes texte avec virgules décimales et espaces milliers."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            s = out[c].astype(str).str.strip()
            # Évite les très longues chaînes (ex. GeoJSON/URL) pour ne pas convertir à tort
            if s.str.len().median() > 50:
                continue
            s2 = (
                s.str.replace("\u00a0", " ", regex=False)  # NBSP
                 .str.replace(" ", "", regex=False)        # espaces milliers
                 .str.replace(",", ".", regex=False)       # virgule -> point
            )
            conv = pd.to_numeric(s2, errors="coerce")
            if conv.notna().mean() >= 0.7:                # conversion seulement si >=70% OK
                out[c] = conv
    return out

@st.cache_data(show_spinner=False)
def load_csv_robust(file_like, manual_sep: str | None = None) -> pd.DataFrame:
    """Lecture robuste : détection séparateur + encodage + quotes + fallback + nettoyage + conversion numérique."""
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

    # Retente avec des séparateurs alternatifs si besoin
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
    """Liste les colonnes numériques détectées."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def get_cat_cols(df, max_uniques=100):
    """Liste les colonnes catégorielles (non numériques) avec un nombre de modalités raisonnable."""
    cats = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) <= max_uniques:
            cats.append(c)
    return cats

# =========================
#         UI IMPORT
# =========================

st.title("📊 CSV Dashboard robuste")
# Panneau latéral = import de fichier + option pour forcer le séparateur
with st.sidebar:
    st.header("⚙️ Import")
    manual_toggle = st.toggle("Forcer un séparateur", value=False)
    manual_sep = None
    if manual_toggle:
        manual_sep = st.radio("Séparateur", options=[";", ",", "\\t", "|"], index=0, horizontal=True)
        if manual_sep == "\\t":
            manual_sep = "\t"
    st.caption("Astuce : beaucoup de CSV européens utilisent `;`.")

# Chargement du CSV
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

# Détection des types de colonnes
num_cols = get_num_cols(df)
cat_cols = get_cat_cols(df)

# Petits indicateurs de base
k1, k2, k3 = st.columns(3)
k1.metric("Lignes", f"{len(df):,}".replace(",", " "))
k2.metric("Colonnes", f"{df.shape[1]}")
k3.metric("Numériques", f"{len(num_cols)}")

# =========================
#          ONGLETs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Barres", "🟢 Nuage de point", "🥧 Camembert", "📈 Linéaire"])

# ------------ ONGLET 1 : BARRES ------------
with tab1:
    st.subheader("Barres")
    st.caption("Comparer des mesures par catégorie. Utilise un 'Top N' pour garder le graphe lisible.")
    mode = st.radio("Mode", ["Agrégation (1 mesure)", "Comparatif (2 mesures)"], horizontal=True)

    # Messages d’aide selon le mode choisi
    if mode == "Agrégation (1 mesure)":
        if not num_cols or not cat_cols:
            st.info("Besoin d’au moins 1 numérique + 1 catégorielle.")
    else:
        if len(num_cols) < 2 or not cat_cols:
            st.info("Besoin d’au moins 1 catégorielle + 2 numériques.")

    # --- Barres agrégées ---
    if mode == "Agrégation (1 mesure)" and num_cols and cat_cols:
        g = st.selectbox("Grouper par (catégoriel)", cat_cols, key="bar_g")
        y = st.selectbox("Mesure (numérique)", num_cols, key="bar_y")
        how = st.selectbox("Agrégation", ["sum", "mean", "median", "min", "max"], index=0, key="bar_how")
        topn = st.slider("Top catégories à afficher", 3, 50, 12, key="bar_topn")
        df_bar = df.groupby(g, dropna=False)[y].agg(how).sort_values(ascending=False).head(topn)

        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(df_bar.index.astype(str), df_bar.values)
        ax.set_xlabel(g); ax.set_ylabel(y); ax.set_title(f"{how} de {y} par {g}")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)

    # --- Barres comparatives (2 mesures) ---
    if mode == "Comparatif (2 mesures)" and len(num_cols) >= 2 and cat_cols:
        g = st.selectbox("Catégorie (X)", cat_cols, key="cmp_g")
        y1 = st.selectbox("Mesure 1", num_cols, key="cmp_y1")
        y2 = st.selectbox("Mesure 2", [c for c in num_cols if c != y1], key="cmp_y2")
        how = st.selectbox("Agrégation", ["sum", "mean", "median", "min", "max"], index=0, key="cmp_how")

        base = df.groupby(g, dropna=False)[[y1, y2]].agg(how)
        # Tri par la valeur max entre y1 et y2 (catégories les plus "importantes" en tête)
        base["__tri__"] = base[[y1, y2]].max(axis=1)
        base = base.sort_values("__tri__", ascending=False).drop(columns="__tri__")
        topn = st.slider("Top catégories à afficher", 3, min(50, len(base)), min(12, len(base)))
        base = base.head(topn)

        # Barres côte-à-côte
        idx = np.arange(len(base))
        width = 0.42
        fig, ax = plt.subplots(figsize=(10,5))
        ax.bar(idx - width/2, base[y1].values, width, label=y1)
        ax.bar(idx + width/2, base[y2].values, width, label=y2)
        ax.set_xticks(idx, base.index.astype(str), rotation=30, ha="right")
        ax.set_xlabel(g); ax.set_ylabel("Valeur"); ax.set_title(f"{how} de {y1} & {y2} par {g}")
        ax.legend()
        st.pyplot(fig)

        

# ------------ ONGLET 2 : NUAGE DE POINT ------------
with tab2:
    st.subheader("Nuage de point (X et Y numériques)")
    st.caption("Visualiser la relation entre deux variables numériques. Limite aux 'Top N' points (triés sur Y).")
    if len(num_cols) < 2:
        st.info("Besoin d’au moins 2 colonnes numériques.")
    else:
        x = st.selectbox("Axe X (num.)", num_cols, key="sc_x")
        y = st.selectbox("Axe Y (num.)", [c for c in num_cols if c != x], key="sc_y")
        d = df[[x, y]].dropna()
        # Top N points selon Y décroissant (plus marquants en premier)
        topn = st.slider("Top points à afficher", 10, min(500, len(d)), min(100, len(d)))
        d = d.sort_values(y, ascending=False).head(topn)

        fig, ax = plt.subplots(figsize=(7,4))
        ax.scatter(d[x], d[y], alpha=0.85)
        ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"{y} vs {x}")
        st.pyplot(fig)

# ------------ ONGLET 3 : CAMEMBERT (comptage simple) ------------
with tab3:
    st.subheader("Répartition (camembert)")
    st.caption("Choisir une colonne catégorielle. Affiche la répartition (comptage) des 'Top N' modalités.")
    if not cat_cols:
        st.info("Besoin d’au moins 1 catégorielle.")
    else:
        c = st.selectbox("Catégorie", cat_cols, key="pie_c")
        series = df[c].astype(str).value_counts()
        # Top N modalités les plus fréquentes
        topn = st.slider("Top catégories à afficher", 3, min(30, len(series)), min(12, len(series)))
        series = series.head(topn)

        fig, ax = plt.subplots(figsize=(6,6))
        ax.pie(series.values, labels=series.index, autopct="%.1f%%", startangle=90)
        ax.set_title(f"Répartition de {c} (Top {topn})")
        st.pyplot(fig)

# ------------ ONGLET 4 : LINEAIRE (simplifié) ------------
with tab4:
    st.subheader("Courbes (linéaire)")
    st.caption("Trace une ou plusieurs séries numériques Y en fonction de X. "
               "Le 'Top N' garde les X les plus significatifs.")
    xcol = st.selectbox("Axe X", df.columns, key="ln_x")
    ycols = st.multiselect("Séries Y (numériques)", num_cols, default=num_cols[:1], key="ln_y")

    if not ycols:
        st.info("Choisis au moins une série Y.")
    else:
        d = df[[xcol] + ycols].dropna()

        # Si X est catégoriel/texte : on agrège par X (somme) puis on prend les Top N catégories
        if not pd.api.types.is_numeric_dtype(d[xcol]):
            agg = d.groupby(xcol, dropna=False)[ycols].sum()
            # Score pour trier les catégories = maximum parmi les Y sélectionnés
            score = agg[ycols].max(axis=1)
            agg = agg.loc[score.sort_values(ascending=False).index]
            topn = st.slider("Top catégories à afficher", 3, min(50, len(agg)), min(12, len(agg)))
            agg = agg.head(topn)

            fig, ax = plt.subplots(figsize=(9,4))
            for c in ycols:
                ax.plot(agg.index.astype(str), agg[c], label=c)
            ax.set_xlabel(xcol); ax.set_ylabel("Valeur"); ax.set_title("Courbe(s) linéaire(s) par catégorie")
            plt.xticks(rotation=30, ha="right")
            ax.legend()
            st.pyplot(fig)

        else:
            # X numérique : on trie par X croissant, puis on garde Top N points les plus "grands" (score = max des Y)
            d = d.sort_values(xcol)
            score = d[ycols].max(axis=1)
            d = d.loc[score.sort_values(ascending=False).index]
            topn = st.slider("Top points à afficher", 10, min(500, len(d)), min(200, len(d)))
            d = d.head(topn).sort_values(xcol)  # ré-ordonne l’affichage sur X

            fig, ax = plt.subplots(figsize=(9,4))
            for c in ycols:
                ax.plot(d[xcol], d[c], label=c)
            ax.set_xlabel(xcol); ax.set_ylabel("Valeur"); ax.set_title("Courbe(s) linéaire(s)")
            ax.legend()
            st.pyplot(fig)
