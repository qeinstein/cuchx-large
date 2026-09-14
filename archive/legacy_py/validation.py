import pandas as pd
from sklearn.model_selection import KFold
import os


def create_subject_splits(csv_path="training_qa.csv", n_splits=5, output_dir="splits"):
    """
    Creates rigorous cross-subject validation splits.
    The true Kaggle test set evaluates on unseen subjects.
    Therefore, our local validation must NEVER train on the validation subjects.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Extract subject ID from path.
    # HAU format: HAU/user1/1-1-1 (subject is index 1)
    # HARn format: HARn/action/user1/trial (subject is index 2)
    def extract_subject(path):
        parts = str(path).split("/")
        if parts[0] == "HAU" and len(parts) > 1:
            return parts[1]
        elif parts[0] == "HARn" and len(parts) > 2:
            return parts[2]
        return "unknown"

    df["subject_id"] = df["path"].apply(extract_subject)

    unique_subjects = df["subject_id"].unique()
    print(f"Found {len(unique_subjects)} unique subjects: {unique_subjects}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_subj_idx, val_subj_idx) in enumerate(kf.split(unique_subjects)):
        train_subjects = unique_subjects[train_subj_idx]
        val_subjects = unique_subjects[val_subj_idx]

        train_df = df[df["subject_id"].isin(train_subjects)]
        val_df = df[df["subject_id"].isin(val_subjects)]

        train_df.to_csv(f"{output_dir}/fold_{fold}_train.csv", index=False)
        val_df.to_csv(f"{output_dir}/fold_{fold}_val.csv", index=False)

        print(f"Fold {fold}: Train={len(train_df)} samples, Val={len(val_df)} samples")


if __name__ == "__main__":
    create_subject_splits()
