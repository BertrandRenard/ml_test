#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

DATE_COL = "Date"
ID_COL = "Id"  # ignoré
TARGET = "log_CDS_5Y"

CAT_COLS = ["Main_FO_Rating", "Main_BusinessArea", "Main_RiskCountry", "Currency"]

# colonnes macro possibles (sensible aux noms exacts vus dans tes logs)
MACRO_CANDIDATES = [
    "Inflation - CPI norm",
    "labour - unemployement rate norm",
    "industrial production norm",
    "Government budget balance norm",
    "Money supply M1 norm",
    "Money supply M2 norm",
    "Consumer confidence norm",
]

def hr():
    print("-" * 120)

def load_data():
    print("Chargement…")
    X_train = pd.read_csv("X_train_dku.csv", low_memory=False)
    y_train = pd.read_csv("y_reg_train_dku.csv", low_memory=False)
    X_test  = pd.read_csv("X_test_dku.csv", low_memory=False)

    # Dates
    for df, nm in [(X_train, "X_train"), (X_test, "X_test")]:
        if DATE_COL in df.columns:
            df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
            print(f"[{nm}] NaT sur {DATE_COL}: {df[DATE_COL].isna().sum()}")

    # Merge cible dans train
    train = X_train.merge(y_train, on=ID_COL, how="left", validate="1:1")

    print(f"train shape={train.shape} | test shape={X_test.shape}")
    return train, X_test

def coverage(df: pd.DataFrame, name: str):
    hr()
    print(f"[COUVERTURE TEMPORELLE] {name}")
    if DATE_COL not in df.columns:
        print("Colonne Date absente."); return
    print("min:", df[DATE_COL].min(), " | max:", df[DATE_COL].max())
    by_year = df[DATE_COL].dt.year.value_counts().sort_index()
    print("Répartition par année:")
    print(by_year.to_string())

    by_month = df[DATE_COL].dt.to_period("M").value_counts().sort_index()
    print("\nTop 10 mois les plus représentés:")
    print(by_month.sort_values(ascending=False).head(10).to_string())

def compare_train_test_over_time(train: pd.DataFrame, test: pd.DataFrame):
    hr()
    print("[TRAIN vs TEST] Comptes par année & par mois")
    for freq, label in [("Y", "Année"), ("M", "Mois")]:
        tr = train[DATE_COL].dt.to_period(freq).value_counts().sort_index()
        te = test [DATE_COL].dt.to_period(freq).value_counts().sort_index()
        df = pd.DataFrame({"train": tr, "test": te}).fillna(0).astype(int)
        print(f"\n{label}:")
        print(df.tail(24).to_string())

def target_time_aggregates(train: pd.DataFrame):
    hr()
    print("[CIBLE] Agrégats temporels globaux")
    if TARGET not in train.columns:
        print("Cible absente."); return
    tmp = train[[DATE_COL, TARGET]].dropna().copy()
    tmp["month"] = tmp[DATE_COL].dt.to_period("M")

    monthly = tmp.groupby("month")[TARGET].agg(["count","mean","median","std","min","max"])
    yearly  = tmp.groupby(tmp[DATE_COL].dt.year)[TARGET].agg(["count","mean","median","std","min","max"])

    print("\nPar MOIS (derniers 24):")
    print(monthly.tail(24).to_string())
    print("\nPar ANNÉE:")
    print(yearly.to_string())

    # Mois avec rupture locale (variation absolue de la moyenne m/m)
    m = monthly["mean"].astype(float)
    delta = (m - m.shift(1)).abs()
    print("\nTop 10 variations absolues de moyenne m/m (changements potentiels de régime):")
    print(delta.sort_values(ascending=False).head(10).to_string())

def temporal_ranks_stability(train: pd.DataFrame, col: str):
    if col not in train.columns or TARGET not in train.columns:
        return
    hr()
    print(f"[STABILITÉ DE CLASSEMENT] {col} — Spearman année t vs t+1")
    df = train[[DATE_COL, col, TARGET]].dropna().copy()
    df["year"] = df[DATE_COL].dt.year
    # moyenne cible par année x catégorie
    pivot = df.groupby(["year", col])[TARGET].mean().reset_index()
    years = sorted(pivot["year"].unique())
    rows = []
    for t, t1 in zip(years, years[1:]):
        a = pivot[pivot["year"]==t].set_index(col)[TARGET]
        b = pivot[pivot["year"]==t1].set_index(col)[TARGET]
        common = a.index.intersection(b.index)
        if len(common) < 3:
            continue
        rho = a.loc[common].rank().corr(b.loc[common].rank(), method="spearman")
        rows.append((t, t1, len(common), rho))
    if rows:
        out = pd.DataFrame(rows, columns=["year_t","year_t1","n_common","spearman_rank_corr"])
        print(out.to_string(index=False))
        print("\nMoyenne corrélation de rangs:", out["spearman_rank_corr"].mean())
    else:
        print("Pas assez de catégories communes entre années consécutives.")

def macro_lagged_correlations(train: pd.DataFrame):
    hr()
    print("[MACRO] Corrélations cible (moyenne globale) vs macro (lags 0–6 mois)")
    # Série cible agrégée globale par mois
    if TARGET not in train.columns:
        print("Cible absente."); return
    tmp = train[[DATE_COL, TARGET] + [c for c in MACRO_CANDIDATES if c in train.columns]].dropna(subset=[DATE_COL]).copy()
    tmp["month"] = tmp[DATE_COL].dt.to_period("M")

    y_m = tmp.groupby("month")[TARGET].mean()

    # macro moyennées par mois (moyenne sur toutes les lignes du mois)
    res_rows = []
    for col in [c for c in MACRO_CANDIDATES if c in tmp.columns]:
        x_m = tmp.groupby("month")[col].mean()
        # align
        df = pd.concat([y_m, x_m], axis=1, join="inner").dropna()
        if df.empty:
            continue
        for lag in range(0, 7):  # macro à t-lag → corrèle avec cible t
            y = df[TARGET]
            x = x_m.reindex(df.index).shift(lag)
            z = pd.concat([y, x], axis=1).dropna()
            if len(z) < 12:
                continue
            r = z[TARGET].corr(z[col])
            res_rows.append((col, lag, len(z), r))
    if res_rows:
        res = pd.DataFrame(res_rows, columns=["macro_col","lag_months","n_pts","pearson_corr"])
        # Top corrélations par macro (en valeur absolue)
        top = res.sort_values(by="pearson_corr", key=lambda s: s.abs(), ascending=False).groupby("macro_col").head(1)
        print("Top corrélations par macro (lag optimal 0–6):")
        print(top.sort_values(by="pearson_corr", key=lambda s: s.abs(), ascending=False).to_string(index=False))
        print("\nAperçu complet (top 25 absolus):")
        print(res.sort_values(by="pearson_corr", key=lambda s: s.abs(), ascending=False).head(25).to_string(index=False))
    else:
        print("Aucune combinaison exploitable.")

def rolling_volatility_of_target(train: pd.DataFrame, window=6):
    hr()
    print(f"[VOLATILITÉ ROLLING] Moyenne globale par mois — fenêtre={window} mois")
    if TARGET not in train.columns:
        print("Cible absente."); return
    tmp = train[[DATE_COL, TARGET]].dropna().copy()
    tmp["month"] = tmp[DATE_COL].dt.to_period("M")
    y_m = tmp.groupby("month")[TARGET].mean().astype(float)
    vol = y_m.pct_change().rolling(window).std()
    tab = pd.DataFrame({"y_mean": y_m, "pct_change": y_m.pct_change(), f"roll_std_{window}m": vol})
    print(tab.tail(24).to_string())
    print("\nTop 10 pics de volatilité rolling:")
    print(vol.sort_values(ascending=False).head(10).to_string())

def feature_temporal_variability(train: pd.DataFrame):
    hr()
    print("[VARIABILITÉ TEMPORELLE DES FEATURES] (hors catégorielles, hors Id, hors cible)")
    num_cols = []
    for c, dt in train.dtypes.items():
        if c in (ID_COL, DATE_COL, TARGET) or (c in CAT_COLS):
            continue
        if pd.api.types.is_numeric_dtype(dt):
            num_cols.append(c)
    if DATE_COL not in train.columns or not num_cols:
        print("Rien à évaluer."); return

    df = train[[DATE_COL] + num_cols].copy()
    df["month"] = df[DATE_COL].dt.to_period("M")

    rows = []
    for c in num_cols:
        # variabilité dans le temps de la moyenne mensuelle et dispersion intra-mois
        m_by_t = df.groupby("month")[c].mean()
        v_time = m_by_t.std()
        intra = df.groupby("month")[c].std().mean()
        rows.append((c, float(v_time), float(intra)))
    out = pd.DataFrame(rows, columns=["feature","std_of_monthly_mean","mean_intra_month_std"])
    # Indice de "sensibilité temporelle": ratio std_temps / (intra + 1e-9)
    out["temporal_sensitivity"] = out["std_of_monthly_mean"] / (out["mean_intra_month_std"] + 1e-9)
    print("Top 20 features sensibles au temps (std_monthly_mean élevé / intra-mois):")
    print(out.sort_values(by="temporal_sensitivity", ascending=False).head(20).to_string(index=False))

def main():
    train, test = load_data()

    # 1) Couverture
    coverage(train, "TRAIN")
    coverage(test , "TEST")

    # 2) Train vs Test sur le temps
    compare_train_test_over_time(train, test)

    # 3) Cible dans le temps (globale)
    target_time_aggregates(train)

    # 4) Stabilité des classements par catégories
    for c in CAT_COLS:
        temporal_ranks_stability(train, c)

    # 5) Macro ↔ cible (lags)
    macro_lagged_correlations(train)

    # 6) Volatilité rolling de la cible (globale)
    rolling_volatility_of_target(train, window=6)

    # 7) Variabilité temporelle des features numériques
    feature_temporal_variability(train)

    hr()
    print("FIN – Analyse temporelle")

if __name__ == "__main__":
    main()
