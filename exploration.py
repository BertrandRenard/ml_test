import sys
import gc
import math
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

# -------------------------
# Réglages
# -------------------------
TOP_K = 30               # nb max de colonnes listées dans certains tableaux
SHOW_TOP_CATS = 10       # nb de modalités les plus fréquentes à afficher
DATE_COL = "Date"        # nom attendu de la date
ID_COL = "Id"            # nom de l'identifiant unique
COMPANY_COL = "Company"  # nom de l'entreprise

# Heuristiques pour types
CAT_HINTS = {"Currency","Main_FO_Rating","Main_BusinessArea","Main_RiskCountry","Company"}
NUM_SUFFIX_HINTS = ("_AnnVal","_QrtVal","_norm","%","_pct","_ratio")
LIKELY_NUM_COLS = {"Close Price","Num share","Mkt Cap","Period Last Market Cap_QrtVal"}

# Règles “sanity check” spécifiques (souples)
RANGE_RULES = {
    # ratio : (min_attendu, max_attendu)
    "Gross Profit Margin_AnnVal": (-1.0, 1.0),
    "Net Income Margin_AnnVal": (-1.0, 1.0),
    "Return On Equity%_AnnVal": (-5.0, 5.0),
    "Return on Assets_AnnVal": (-5.0, 5.0),
    "Total Debt / Total Capital_AnnVal": (0.0, 2.0),
    "Long-Term Debt / Total_AnnVal": (0.0, 2.0),
    "Asset Turnover_QrtVal": (-5.0, 5.0),
    "FFO Interest Coverage_AnnVal": (-10.0, 100.0),
    "EBITDA / Interest Expense_QrtVal": (-10.0, 100.0),
}

def hrule():
    print("-"*120)

def try_parse_date(df: pd.DataFrame, col: str):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def classify_columns(df: pd.DataFrame):
    """Renvoie (num_cols, cat_cols, other_cols) par heuristiques + dtype."""
    cols = list(df.columns)
    num_cols, cat_cols, other_cols = [], [], []
    for c in cols:
        if c == ID_COL or c == DATE_COL: 
            continue
        dt = df[c].dtype
        # heuristiques
        if c in CAT_HINTS:
            cat_cols.append(c)
            continue
        if (isinstance(dt, pd.CategoricalDtype) or
            dt == object and df[c].nunique(dropna=False) <= max(200, int(len(df)*0.01))):
            # faible cardinalité => probablement catégorielle
            cat_cols.append(c)
            continue
        if np.issubdtype(dt, np.number) or any(c.endswith(suf) for suf in NUM_SUFFIX_HINTS) or c in LIKELY_NUM_COLS:
            num_cols.append(c)
        elif dt == object:
            # object volumineux peut être numérique sale
            # si >95% parsable en float -> num
            sample = df[c].dropna().astype(str).head(1000)
            parsable = sample.apply(lambda x: x.replace(",","").replace(" ","").replace("%","").replace("\xa0",""))
            ok = 0
            for v in parsable:
                try:
                    float(v)
                    ok += 1
                except Exception:
                    pass
            if len(sample) > 0 and ok/len(sample) >= 0.95:
                num_cols.append(c)
            else:
                cat_cols.append(c)
        else:
            other_cols.append(c)
    return sorted(set(num_cols)), sorted(set(cat_cols)), sorted(set(other_cols))

def memory_usage(df: pd.DataFrame, name: str):
    mem_mb = df.memory_usage(deep=True).sum()/1024**2
    print(f"[MEM] {name}: {mem_mb:,.2f} MB")

def load_all():
    print("Chargement des fichiers CSV…")
    X_train = pd.read_csv("X_train_dku.csv", low_memory=False)
    y_train  = pd.read_csv("y_reg_train_dku.csv", low_memory=False)
    X_test  = pd.read_csv("X_test_dku.csv", low_memory=False)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train  shape: {y_train.shape}")
    print(f"X_test  shape: {X_test.shape}")
    memory_usage(X_train, "X_train")
    memory_usage(y_train , "y_train")
    memory_usage(X_test , "X_test")

    # parse dates si présent
    for df, nm in [(X_train,"X_train"), (X_test,"X_test")]:
        before_na = df[DATE_COL].isna().sum() if DATE_COL in df.columns else "NA"
        try_parse_date(df, DATE_COL)
        after_na = df[DATE_COL].isna().sum() if DATE_COL in df.columns else "NA"
        if DATE_COL in df.columns:
            print(f"[{nm}] {DATE_COL}: parsed -> NaT count before={before_na}, after={after_na}")

    # merge cible
    if ID_COL not in X_train.columns or ID_COL not in y_train.columns:
        print("[ERREUR] Colonne Id absente. Vérifie les fichiers.")
        sys.exit(2)
    train = X_train.merge(y_train, on=ID_COL, how="left", validate="1:1")
    missing_target = train["log_CDS_5Y"].isna().sum() if "log_CDS_5Y" in train.columns else -1
    print(f"Fusion cible: train shape={train.shape}, valeurs manquantes dans la cible={missing_target}")
    return train, X_test

def basic_integrity_checks(df: pd.DataFrame, name: str):
    hrule()
    print(f"[INTÉGRITÉ] {name}")
    print("Aperçu colonnes / dtypes:")
    print(df.dtypes.sort_index())
    hrule()

    # unicité Id
    if ID_COL in df.columns:
        dup = df.duplicated(ID_COL).sum()
        print(f"Doublons sur {ID_COL}: {dup}")
    else:
        print(f"{ID_COL} absent.")

    # doublons (toutes colonnes)
    dups_all = df.duplicated().sum()
    print(f"Doublons (toutes colonnes) : {dups_all}")

    # cohérence Company-Date
    if COMPANY_COL in df.columns and DATE_COL in df.columns:
        grp = df.groupby([COMPANY_COL, DATE_COL]).size()
        multi = (grp > 1).sum()
        print(f"Lignes multiples par ({COMPANY_COL},{DATE_COL}) : {multi}")
        if multi > 0:
            print("Top 10 couples en doublon :")
            print(grp[grp>1].sort_values(ascending=False).head(10))
    else:
        print("Colonnes pour contrôle (Company/Date) manquantes.")

    # couverture temporelle
    if DATE_COL in df.columns:
        print("Couverture temporelle :")
        print(" - min date :", df[DATE_COL].min())
        print(" - max date :", df[DATE_COL].max())
        by_year = df[DATE_COL].dt.year.value_counts().sort_index()
        print("Répartition par année (top 20):")
        print(by_year.head(20))

def missingness_report(df: pd.DataFrame, name: str, topk=TOP_K):
    hrule()
    print(f"[MANQUANTS] {name}")
    na = df.isna().mean().sort_values(ascending=False)
    print("Top colonnes avec valeurs manquantes (taux) :")
    print((na.head(topk)*100).round(2).astype(str) + "%")
    print("Colonnes sans aucun manquant :", int((na==0).sum()), "/", len(na))
    # patterns simples
    print("Nombre de lignes totalement vides (hors Id/Date) :", df.drop(columns=[c for c in [ID_COL,DATE_COL] if c in df.columns]).isna().all(axis=1).sum())

def cardinality_report(df: pd.DataFrame, name: str, cat_cols):
    hrule()
    print(f"[CARDINALITÉ CATEGORIELLES] {name}")
    out_lines = []
    for c in cat_cols:
        nun = df[c].nunique(dropna=True)
        nanr = df[c].isna().mean()*100
        out_lines.append((c, nun, f"{nanr:.2f}%"))
    if out_lines:
        res = pd.DataFrame(out_lines, columns=["col","n_unique","%NaN"]).sort_values("n_unique", ascending=False)
        print(res.head(TOP_K).to_string(index=False))
    else:
        print("Aucune colonne catégorielle détectée.")

    # top modalités
    for c in cat_cols[:10]:
        print(f"\nTop modalités - {c}:")
        print(df[c].value_counts(dropna=False).head(SHOW_TOP_CATS))

def numeric_summary(df: pd.DataFrame, name: str, num_cols):
    hrule()
    print(f"[RÉSUMÉ NUMÉRIQUES] {name}")
    stats = []
    for c in num_cols[:]:
        s = df[c]
        # ignorer séries entièrement vides
        if s.notna().sum() == 0:
            stats.append([c, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "100.00%", "NA", "NA"])
            continue
        v = s.replace([np.inf,-np.inf], np.nan)
        q1 = v.quantile(0.01)
        q99 = v.quantile(0.99)
        stats.append([
            c,
            int(v.count()),
            v.nunique(dropna=True),
            float(v.min(skipna=True)),
            float(v.max(skipna=True)),
            float(v.mean(skipna=True)),
            float(v.std(skipna=True)),
            float(q1) if pd.notna(q1) else np.nan,
            float(q99) if pd.notna(q99) else np.nan,
            f"{(v.eq(0).mean()*100):.2f}%",
            f"{(v.lt(0).mean()*100):.2f}%",
            f"{(v.isna().mean()*100):.2f}%"
        ])
    cols = ["col","count","n_unique","min","max","mean","std","p01","p99","%zero","%neg","%NaN"]
    dfstats = pd.DataFrame(stats, columns=cols)
    # ordonner : %NaN desc puis std desc
    dfstats["_nan"] = dfstats["%NaN"].str.rstrip("%").astype(float)
    dfstats = dfstats.sort_values(by=["_nan","std"], ascending=[False, False]).drop(columns=["_nan"])
    print(dfstats.head(TOP_K).to_string(index=False))

def target_report(train: pd.DataFrame):
    hrule()
    print("[CIBLE] log_CDS_5Y (train)")
    if "log_CDS_5Y" not in train.columns:
        print("Cible absente du train fusionné.")
        return
    s = train["log_CDS_5Y"].replace([np.inf,-np.inf], np.nan)
    print(f"Non-nuls : {s.notna().sum()} / {len(s)} ; NaN : {s.isna().sum()}")
    print(f"Min={s.min():.6f} | P1={s.quantile(0.01):.6f} | P50={s.median():.6f} | P99={s.quantile(0.99):.6f} | Max={s.max():.6f} | Mean={s.mean():.6f} | Std={s.std():.6f}")

def range_sanity_checks(df: pd.DataFrame, name: str):
    hrule()
    print(f"[SANITY CHECKS DE PLAGES] {name}")
    for col, (lo, hi) in RANGE_RULES.items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            n = len(s)
            below = (s < lo).sum()
            above = (s > hi).sum()
            nan = s.isna().sum()
            print(f"{col}: [{lo}, {hi}] | below={below} ({below/n:.2%}) | above={above} ({above/n:.2%}) | NaN={nan} ({nan/n:.2%})")

def compare_train_test_levels(train: pd.DataFrame, test: pd.DataFrame, cat_cols):
    hrule()
    print("[COMPARAISON TRAIN vs TEST - Catégorielles]")
    for c in cat_cols[:]:
        tr = set(train[c].dropna().astype(str).unique()) if c in train.columns else set()
        te = set(test[c].dropna().astype(str).unique()) if c in test.columns else set()
        new_in_test = te - tr
        missing_in_test = tr - te
        print(f"{c}: train#={len(tr)} | test#={len(te)} | nouvelles_mod={len(new_in_test)} | absentes_dans_test={len(missing_in_test)}")
        if new_in_test:
            print(f"  -> Exemples nouvelles (test pas dans train): {list(sorted(new_in_test))[:SHOW_TOP_CATS]}")
        if missing_in_test:
            print(f"  -> Exemples absentes (train pas dans test): {list(sorted(missing_in_test))[:SHOW_TOP_CATS]}")

def missing_by_group(df: pd.DataFrame, group_col: str, cols: list, name: str, topk=10):
    if group_col not in df.columns:
        return
    hrule()
    print(f"[MANQUANTS PAR GROUPE] {name} – groupe={group_col}")
    res = []
    for c in cols:
        m = df[c].isna().mean()
        if m == 0:
            continue
        g = df.groupby(group_col)[c].apply(lambda s: s.isna().mean()).sort_values(ascending=False)
        res.append((c, g.head(topk)))
    for c, series in res:
        print(f"\nColonne: {c} – Top {topk} groupes avec plus de NaN:")
        print((series*100).round(2).astype(str) + "%")

def per_company_frequency(df: pd.DataFrame, name: str):
    if COMPANY_COL not in df.columns or DATE_COL not in df.columns:
        return
    hrule()
    print(f"[FRÉQUENCE PAR SOCIÉTÉ] {name}")
    counts = df.groupby(COMPANY_COL)[DATE_COL].nunique().describe()
    print("Nb de périodes distinctes par Company (statistiques) :")
    print(counts.to_string())
    print("Top 10 sociétés avec le plus de périodes manquantes (en supposant mensuel) :")
    # estimation “trous” : comparer span temporel et nb points
    gaps = []
    for comp, g in df[[COMPANY_COL, DATE_COL]].dropna().groupby(COMPANY_COL):
        if g[DATE_COL].empty: 
            continue
        span = (g[DATE_COL].max() - g[DATE_COL].min()).days/30.44 + 1e-9
        expected = int(round(span)) + 1
        observed = g[DATE_COL].nunique()
        gaps.append((comp, expected - observed))
    if gaps:
        gaps = sorted(gaps, key=lambda x: x[1], reverse=True)[:10]
        print(pd.DataFrame(gaps, columns=["Company","trous_estimes"]).to_string(index=False))

def numeric_vs_target_corr(train: pd.DataFrame, num_cols: list):
    if "log_CDS_5Y" not in train.columns:
        return
    hrule()
    print("[CORRÉLATION SPEARMAN vs CIBLE] (numériques)")
    corr = {}
    y = train["log_CDS_5Y"]
    for c in num_cols:
        try:
            rho = train[[c]].assign(y=y).corr(method="spearman").iloc[0,1]
            corr[c] = rho
        except Exception:
            pass
    corr = pd.Series(corr).dropna().sort_values(key=lambda s: s.abs(), ascending=False)
    print(corr.head(TOP_K))

def main():
    train, test = load_all()

    # rapports d’intégrité généraux
    basic_integrity_checks(train, "TRAIN (X+y)")
    basic_integrity_checks(test, "TEST (X)")

    # classification colonnes
    num_cols_tr, cat_cols_tr, other_tr = classify_columns(train)
    num_cols_te, cat_cols_te, other_te = classify_columns(test)

    hrule()
    print("[CLASSIFICATION COLONNES] TRAIN")
    print("Numériques (top):", num_cols_tr[:TOP_K])
    print("Catégorielles (top):", cat_cols_tr[:TOP_K])
    print("Autres:", other_tr)
    hrule()
    print("[CLASSIFICATION COLONNES] TEST")
    print("Numériques (top):", num_cols_te[:TOP_K])
    print("Catégorielles (top):", cat_cols_te[:TOP_K])
    print("Autres:", other_te)

    # cohérence colonnes train/test
    hrule()
    print("[COHÉRENCE COLONNES TRAIN vs TEST]")
    cols_tr = set(train.columns) - {"log_CDS_5Y"}
    cols_te = set(test.columns)
    only_tr = sorted(list(cols_tr - cols_te))
    only_te = sorted(list(cols_te - cols_tr))
    print("Colonnes seulement dans TRAIN (hors cible):", only_tr[:TOP_K])
    print("Colonnes seulement dans TEST:", only_te[:TOP_K])

    # manquants
    missingness_report(train, "TRAIN")
    missingness_report(test, "TEST")

    # cardinalité catégorielles
    # (prendre l’union détectée sur train/test)
    cat_union = sorted(set(cat_cols_tr).union(cat_cols_te))
    cardinality_report(train, "TRAIN", cat_union)
    cardinality_report(test , "TEST" , cat_union)

    # numériques – résumés
    num_union = sorted(set(num_cols_tr).union(num_cols_te))
    numeric_summary(train, "TRAIN", num_union)
    numeric_summary(test , "TEST" , num_union)

    # cible
    target_report(train)

    # sanity checks métiers
    range_sanity_checks(train, "TRAIN")
    range_sanity_checks(test , "TEST")

    # comparaison des niveaux des catégorielles
    compare_train_test_levels(train, test, cat_union)

    # manquants par groupe (si colonnes présentes)
    for grp in ["Main_RiskCountry","Main_BusinessArea","Currency", COMPANY_COL]:
        missing_by_group(train, grp, num_union[:TOP_K] + cat_union[:TOP_K], f"TRAIN")
        missing_by_group(test , grp, num_union[:TOP_K] + cat_union[:TOP_K], f"TEST")

    # fréquences par société
    per_company_frequency(train, "TRAIN")
    per_company_frequency(test , "TEST")

    # corrélation numériques vs cible
    numeric_vs_target_corr(train, num_union)

    hrule()
    print("FIN DU POINT 1 – Tous les logs sont affichés ci-dessus.")
    hrule()

if __name__ == "__main__":
    main()
