import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def split_users(df, cluster_label_col="cluster", min_reps=20):
    reps = df.groupby(["userId", cluster_label_col]).size().reset_index(name="total_reps")
    continued_pairs = reps[reps["total_reps"] >= min_reps]
    continued_users = set(continued_pairs["userId"])
    all_users = set(df["userId"])
    labels = pd.DataFrame(
        {"userId": list(all_users), "label": [1 if u in continued_users else 0 for u in all_users]}
    ).set_index("userId")
    return labels


def compute_features(df, cluster_label_col="cluster", lookback_reps=10):
    feats = []
    pop = df[cluster_label_col].value_counts().to_dict()
    year_stats = df.groupby(cluster_label_col)["release_year"].agg(["mean", "std"]).to_dict("index")
    for u, sub in df.groupby("userId"):
        sub = sub.sort_values("timestamp").copy()
        sub["rank"] = sub.groupby(cluster_label_col).cumcount() + 1
        sub = sub[sub["rank"] <= lookback_reps]
        p = sub[cluster_label_col].value_counts(normalize=True)
        entropy = -(p * np.log2(p + 1e-9)).sum()
        intervals = (
            sub.sort_values("timestamp")
            .groupby(cluster_label_col)["timestamp"]
            .diff()
            .dropna()
            .dt.total_seconds()
        )
        avg_interval = intervals.mean() if not intervals.empty else 0
        burstiness = intervals.std() / intervals.mean() if intervals.mean() > 0 else 0
        clusters = sub[cluster_label_col].unique()
        pop_mean = np.mean([pop[c] for c in clusters]) if len(clusters) > 0 else 0
        nov_mean = np.mean([year_stats[c]["std"] for c in clusters]) if len(clusters) > 0 else 0
        gender = sub["gender"].iloc[0] if "gender" in sub.columns else np.nan
        age = sub["age"].iloc[0] if "age" in sub.columns else np.nan
        mobility = sub[cluster_label_col].nunique()
        feats.append(
            {
                "userId": u,
                "entropy": entropy,
                "avg_interval": avg_interval,
                "burstiness": burstiness,
                "popularity": pop_mean,
                "novelty": nov_mean,
                "gender": gender,
                "age": age,
                "mobility": mobility,
            }
        )
    feat_df = pd.DataFrame(feats).set_index("userId").fillna(0)
    return feat_df


def run_survival_analysis(df, labels, features, corr_threshold=0.9, penalizer=0.1):
    survival = []
    for u, row in labels.iterrows():
        cont = row["label"]
        sub = df[df["userId"] == u]
        max_rep = sub["cluster_exposure"].max()
        event = 1 - cont
        survival.append({"userId": u, "duration": max_rep, "event": event})
    surv_df = pd.DataFrame(survival).set_index("userId")
    data = surv_df.join(features)
    if "gender" in data.columns:
        data = pd.get_dummies(data, columns=["gender"], drop_first=True)
    num_cols = [c for c in data.columns if c not in ["duration", "event"]]
    scaler = StandardScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(data, duration_col="duration", event_col="event")
    print(cph.summary)
    return cph


def run_classification(features, labels, cv_folds=5):
    df = features.join(labels, how="inner")
    print("Label distribution:\n", df["label"].value_counts())
    numeric = df.drop(columns=["label"]).select_dtypes(include=[np.number])
    print("Feature variances:\n", numeric.var())
    if "gender" in df.columns:
        df = pd.get_dummies(df, columns=["gender"], drop_first=True)
    X = df.drop(columns=["label"]).copy()
    y = df["label"].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []
    importances = np.zeros(X.shape[1])
    for train_idx, test_idx in skf.split(X, y):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        X_train.loc[:, num_cols] = scaler.fit_transform(X_train.loc[:, num_cols])
        X_test.loc[:, num_cols] = scaler.transform(X_test.loc[:, num_cols])
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        cv_scores.append(clf.score(X_test, y_test))
        importances += clf.feature_importances_
    importances /= cv_folds
    print("Classification CV scores:", cv_scores)
    for feat, imp in zip(X.columns, importances):
        print(f"{feat}: {imp:.4f}")
    X_scaled = X.copy()
    X_scaled.loc[:, num_cols] = scaler.fit_transform(X_scaled.loc[:, num_cols])
    final_clf = RandomForestClassifier(random_state=42)
    final_clf.fit(X_scaled, y)
    final_importances = dict(zip(X_scaled.columns, final_clf.feature_importances_))
    return final_clf, cv_scores, final_importances
