import sys
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

TOP_K = 30
SHOW_TOP_CATS = 10
DATE_COL = "Date"
ID_COL = "Id"                 # ignoré dans toutes les analyses
COMPANY_COL = "Company"       # ignoré (sur train et test)

CAT_HINTS = {"Currency","Main_FO_Rating","Main_BusinessArea","Main_RiskCountry","Company"}
NUM_SUFFIX_HINTS = ("_AnnVal","_QrtVal","_norm","%","_pct","_ratio")
LIKELY_NUM_COLS = {"Close Price","Num share","Mkt Cap","Period Last Market Cap_QrtVal"}

RANGE_RULES = {
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

def classify_columns(df: pd.DataFrame):
    """Renvoie (num_cols, cat_cols, other_cols) — ne considère jamais Id/Date."""
    num_cols, cat_cols, other_cols = [], [], []
    for c in df.columns:
        if c in (ID_COL, DATE_COL):  # exclus
            continue
        dt = df[c].dtype
        if c in CAT_HINTS:
            cat_cols.append(c); continue
        if (pd.api.types.is_categorical_dtype(dt) or
            (dt == object and df[c].nunique(dropna=False) <= max(200, int(len(df)*0.01)))):
            cat_cols.append(c); continue
        if pd.api.types.is_numeric_dtype(dt) or any(c.endswith(s) for s in NUM_SUFFIX_HINTS) or c in LIKELY_NUM_COLS:
            num_cols.append(c)
        elif dt == object:
            sample = df[c].dropna().astype(str).head(1000)
            parsable = sample.str.replace(",","", regex=False)\
                             .str.replace(" ","", regex=False)\
                             .str.replace("%","", regex=False)\
                             .str.replace("\xa0","", regex=False)
            ok = 0
            for v in parsable:
                try: float(v); ok += 1
                except: pass
            if len(sample)>0 and ok/len(sample) >= 0.95:
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
    print("Chargement…")
    X_train = pd.read_csv("X_train_dku.csv", low_memory=False)
    y_train  = pd.read_csv("y_reg_train_dku.csv", low_memory=False)
    X_test  = pd.read_csv("X_test_dku.csv", low_memory=False)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train  shape: {y_train.shape}")
    print(f"X_test   shape: {X_test.shape}")
    memory_usage(X_train, "X_train")
    memory_usage(y_train , "y_train")
    memory_usage(X_test  , "X_test")

    try_parse_date(X_train, DATE_COL)
    try_parse_date(X_test , DATE_COL)
    if DATE_COL in X_train.columns:
        print(f"[X_train] {DATE_COL}: NaT={X_train[DATE_COL].isna().sum()}")
    if DATE_COL in X_test.columns:
        print(f"[X_test ] {DATE_COL}: NaT={X_test[DATE_COL].isna().sum()}")

    # Merge cible
    if ID_COL not in X_train.columns or ID_COL not in y_train.columns:
        print("[ERREUR] Colonne Id absente."); sys.exit(2)
    train = X_train.merge(y_train, on=ID_COL, how="left", validate="1:1")
    miss_tgt = train["log_CDS_5Y"].isna().sum() if "log_CDS_5Y" in train.columns else -1
    print(f"Fusion cible: train shape={train.shape}, NaN cible={miss_tgt}")
    return train, X_test

def basic_integrity_checks(df: pd.DataFrame, name: str):
    hrule()
    print(f"[INTÉGRITÉ] {name}")
    print("dtypes:")
    print(df.dtypes.sort_index())
    hrule()
    # Pas de vérif d'unicité/doublons sur Id (exigence)
    # Doublons complets (toutes colonnes) — utile mais neutre vis-à-vis d'Id
    dups_all = df.duplicated().sum()
    print(f"Doublons exacts (toutes colonnes) : {dups_all}")
    # Couverture temporelle
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
    na = df.drop(columns=[c for c in (ID_COL,) if c in df.columns]).isna().mean().sort_values(ascending=False)
    print("Top colonnes avec valeurs manquantes (taux) :")
    print((na.head(topk)*100).round(2).astype(str) + "%")
    print("Colonnes sans aucun manquant :", int((na==0).sum()), "/", len(na))
    core = df.drop(columns=[c for c in (ID_COL, DATE_COL) if c in df.columns])
    print("Lignes totalement vides (hors Id/Date) :", core.isna().all(axis=1).sum())

def cardinality_report(df: pd.DataFrame, name: str, cat_cols):
    hrule()
    print(f"[CARDINALITÉ CATEGORIELLES] {name}")
    cat_cols = [c for c in cat_cols if c in df.columns and c not in (ID_COL,)]
    if not cat_cols:
        print("Aucune colonne catégorielle détectée."); return
    rows = []
    for c in cat_cols:
        rows.append((c, df[c].nunique(dropna=True), f"{df[c].isna().mean()*100:.2f}%"))
    res = pd.DataFrame(rows, columns=["col","n_unique","%NaN"]).sort_values("n_unique", ascending=False)
    print(res.head(TOP_K).to_string(index=False))
    for c in [x for x in cat_cols if x != COMPANY_COL][:10]:
        print(f"\nTop modalités - {c}:")
        print(df[c].value_counts(dropna=False).head(SHOW_TOP_CATS))

def numeric_summary(df: pd.DataFrame, name: str, num_cols):
    hrule()
    print(f"[RÉSUMÉ NUMÉRIQUES] {name}")
    use_cols = [c for c in num_cols if c in df.columns and c not in (ID_COL,)]
    if not use_cols:
        print("Aucune colonne numérique à résumer."); return
    stats = []
    for c in use_cols:
        v = pd.to_numeric(df[c], errors="coerce").replace([np.inf,-np.inf], np.nan)
        cnt = int(v.count())
        if cnt == 0:
            stats.append([c,0,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,"100.00%","0.00%"]); continue
        q1 = v.quantile(0.01); q99 = v.quantile(0.99)
        stats.append([
            c, cnt, float(v.min()), float(v.max()),
            float(v.mean()), float(v.std()),
            float(q1) if pd.notna(q1) else np.nan,
            float(q99) if pd.notna(q99) else np.nan,
            f"{(v.isna().mean()*100):.2f}%",
            f"{(v.eq(0).mean()*100):.2f}%"
        ])
    cols = ["col","count","min","max","mean","std","p01","p99","%NaN","%zero"]
    dfstats = pd.DataFrame(stats, columns=cols).sort_values(by=["%NaN","std"], ascending=[False, False])
    print(dfstats.head(TOP_K).to_string(index=False))

def target_report(train: pd.DataFrame):
    hrule()
    print("[CIBLE] log_CDS_5Y (train)")
    if "log_CDS_5Y" not in train.columns:
        print("Cible absente."); return
    s = pd.to_numeric(train["log_CDS_5Y"], errors="coerce").replace([np.inf,-np.inf], np.nan)
    print(f"Non-nuls : {s.notna().sum()} / {len(s)} ; NaN : {s.isna().sum()}")
    print(f"Min={s.min():.6f} | P1={s.quantile(0.01):.6f} | P50={s.median():.6f} | P99={s.quantile(0.99):.6f} | Max={s.max():.6f} | Mean={s.mean():.6f} | Std={s.std():.6f}")

def range_sanity_checks(df: pd.DataFrame, name: str):
    hrule()
    print(f"[SANITY CHECKS DE PLAGES] {name}")
    for col, (lo, hi) in RANGE_RULES.items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            n = len(s); below = (s < lo).sum(); above = (s > hi).sum(); nan = s.isna().sum()
            print(f"{col}: [{lo}, {hi}] | below={below} ({below/n:.2%}) | above={above} ({above/n:.2%}) | NaN={nan} ({nan/n:.2%})")

def compare_train_test_levels(train: pd.DataFrame, test: pd.DataFrame, cat_cols):
    hrule()
    print("[COMPARAISON TRAIN vs TEST - Catégorielles]")
    cat_cols = [c for c in cat_cols if c not in (ID_COL, COMPANY_COL)]  # exclut Company + Id
    for c in cat_cols:
        if c not in train.columns and c not in test.columns: 
            continue
        tr = set(train[c].dropna().astype(str).unique()) if c in train.columns else set()
        te = set(test[c].dropna().astype(str).unique()) if c in test.columns else set()
        new_in_test = te - tr
        missing_in_test = tr - te
        print(f"{c}: train#={len(tr)} | test#={len(te)} | nouvelles_mod={len(new_in_test)} | absentes_dans_test={len(missing_in_test)}")
        if new_in_test:
            print(f"  -> Exemples nouvelles (test pas dans train): {list(sorted(new_in_test))[:SHOW_TOP_CATS]}")
        if missing_in_test:
            print(f"  -> Exemples absentes (train pas dans test): {list(sorted(missing_in_test))[:SHOW_TOP_CATS]}")

def numeric_vs_target_corr(train: pd.DataFrame, num_cols: list):
    if "log_CDS_5Y" not in train.columns: 
        return
    hrule()
    print("[CORRÉLATION SPEARMAN vs CIBLE] (numériques)")
    corr = {}
    y = pd.to_numeric(train["log_CDS_5Y"], errors="coerce")
    for c in num_cols:
        if c not in train.columns or c in (ID_COL,):
            continue
        try:
            rho = train[[c]].assign(y=y).corr(method="spearman").iloc[0,1]
            corr[c] = rho
        except Exception:
            pass
    if corr:
        s = pd.Series(corr).dropna().sort_values(key=lambda x: x.abs(), ascending=False)
        print(s.head(TOP_K))
    else:
        print("Aucune corrélation calculable.")

def main():
    train, test = load_all()

    basic_integrity_checks(train, "TRAIN (X+y)")
    basic_integrity_checks(test , "TEST (X)")

    num_tr, cat_tr, _ = classify_columns(train)
    num_te, cat_te, _ = classify_columns(test)

    hrule(); print("[CLASSIFICATION COLONNES] TRAIN")
    print("Numériques (top):", num_tr[:TOP_K])
    print("Catégorielles (top):", cat_tr[:TOP_K])

    hrule(); print("[CLASSIFICATION COLONNES] TEST")
    print("Numériques (top):", num_te[:TOP_K])
    print("Catégorielles (top):", cat_te[:TOP_K])

    hrule(); print("[COHÉRENCE COLONNES TRAIN vs TEST]")
    cols_tr = set(train.columns) - {"log_CDS_5Y"}
    cols_te = set(test.columns)
    only_tr = sorted(list(cols_tr - cols_te))
    only_te = sorted(list(cols_te - cols_tr))
    print("Colonnes seulement dans TRAIN (hors cible):", only_tr[:TOP_K])
    print("Colonnes seulement dans TEST:", only_te[:TOP_K])

    missingness_report(train, "TRAIN")
    missingness_report(test , "TEST")

    cat_union = sorted(set(cat_tr).union(cat_te))
    cardinality_report(train, "TRAIN", cat_union)
    cardinality_report(test , "TEST" , cat_union)

    num_union = sorted(set(num_tr).union(num_te))
    numeric_summary(train, "TRAIN", num_union)
    numeric_summary(test , "TEST" , num_union)   # robuste si certaines colonnes manquent

    target_report(train)
    range_sanity_checks(train, "TRAIN")
    range_sanity_checks(test , "TEST")
    compare_train_test_levels(train, test, cat_union)
    numeric_vs_target_corr(train, num_union)

    hrule(); print("FIN DU POINT 1.")
    hrule()

if __name__ == "__main__":
    main()
