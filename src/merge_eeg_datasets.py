from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(r"C:\Epileptic_seizure\datasets")
KAGGLE_PATH = DATA_DIR / "Epileptic_Seizure_kaggle.csv"
BANGALORE_CANDIDATES = [
    DATA_DIR / "Bangalore_EEG.csv",
    DATA_DIR / "Banglore_EEG.csv",
]
OUTPUT_PATH = DATA_DIR / "final_epileptic_seizure_merge.csv"
RANDOM_STATE = 42


def find_existing_path(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "None of these input files were found: "
        + ", ".join(str(path) for path in paths)
    )


def find_label_column(df: pd.DataFrame) -> str:
    label_candidates = {"label", "target", "class", "y"}
    for col in df.columns:
        if str(col).strip().lower() in label_candidates:
            return col
    raise ValueError(
        "Could not find a label column. Expected one of: Label, Target, class, y."
    )


def feature_sort_key(col: str) -> tuple[int, str]:
    text = str(col)
    digits = "".join(ch for ch in text if ch.isdigit())
    return (int(digits), text) if digits else (10**9, text)


def get_numeric_feature_columns(df: pd.DataFrame, label_col: str) -> list[str]:
    feature_cols: list[str] = []
    for col in df.columns:
        if col == label_col:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            feature_cols.append(col)
    return sorted(feature_cols, key=feature_sort_key)


def encode_labels(labels: pd.Series, dataset_name: str) -> pd.Series:
    normalized = labels.astype(str).str.strip().str.lower()

    text_map = {
        "seizure": 1,
        "epileptic": 1,
        "ictal": 1,
        "yes": 1,
        "true": 1,
        "non-seizure": 0,
        "non seizure": 0,
        "normal": 0,
        "healthy": 0,
        "interictal": 0,
        "no": 0,
        "false": 0,
    }
    mapped = normalized.map(text_map)

    numeric = pd.to_numeric(labels, errors="coerce")
    if mapped.isna().all() and numeric.notna().any():
        if dataset_name.lower() == "uci":
            # UCI/Kaggle uses 1 for seizure and 2-5 for non-seizure.
            return (numeric == 1).astype(int)
        return numeric.map(lambda value: 1 if value == 1 else 0).astype(int)

    mapped = mapped.fillna(numeric.map(lambda value: 1 if value == 1 else 0))
    if mapped.isna().any():
        bad_values = labels[mapped.isna()].drop_duplicates().head(10).tolist()
        raise ValueError(f"Unrecognized labels in {dataset_name}: {bad_values}")

    return mapped.astype(int)


def align_feature_count(features: pd.DataFrame, target_count: int) -> pd.DataFrame:
    if features.shape[1] > target_count:
        features = features.iloc[:, :target_count]
    elif features.shape[1] < target_count:
        for idx in range(features.shape[1] + 1, target_count + 1):
            features[f"pad_{idx}"] = 0.0
    return features


def preprocess_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    data_source: int,
    target_feature_count: int,
) -> pd.DataFrame:
    label_col = find_label_column(df)
    feature_cols = get_numeric_feature_columns(df, label_col)

    labels = encode_labels(df[label_col], dataset_name)
    features = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    features = align_feature_count(features, target_feature_count)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    imputed = imputer.fit_transform(features)
    scaled = scaler.fit_transform(imputed)

    output = pd.DataFrame(
        scaled,
        columns=[f"eeg_{idx:03d}" for idx in range(1, target_feature_count + 1)],
    )
    output["Label"] = labels.to_numpy(dtype=int)
    output["data_source"] = data_source
    return output


def main() -> None:
    bangalore_path = find_existing_path(BANGALORE_CANDIDATES)

    uci_df = pd.read_csv(KAGGLE_PATH)
    beed_df = pd.read_csv(bangalore_path)

    uci_label_col = find_label_column(uci_df)
    beed_label_col = find_label_column(beed_df)
    uci_feature_cols = get_numeric_feature_columns(uci_df, uci_label_col)
    beed_feature_cols = get_numeric_feature_columns(beed_df, beed_label_col)

    # Use the shared window size so neither dataset receives many artificial
    # padded EEG values. With the current files this is 16 columns.
    target_feature_count = min(len(uci_feature_cols), len(beed_feature_cols))

    uci_clean = preprocess_dataset(
        uci_df,
        dataset_name="UCI",
        data_source=0,
        target_feature_count=target_feature_count,
    )
    beed_clean = preprocess_dataset(
        beed_df,
        dataset_name="BEED",
        data_source=1,
        target_feature_count=target_feature_count,
    )

    merged = pd.concat([uci_clean, beed_clean], axis=0, ignore_index=True)
    merged = merged.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"UCI input shape: {uci_df.shape}")
    print(f"BEED input shape: {beed_df.shape}")
    print(f"Target EEG feature count: {target_feature_count}")
    print(f"Final merged shape: {merged.shape}")
    print(f"Label counts: {merged['Label'].value_counts().sort_index().to_dict()}")
    print(f"Data source counts: {merged['data_source'].value_counts().sort_index().to_dict()}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
