import pandas as pd


def create_submission(
    predictions_dict, test_csv="test_qa.csv", output_file="submission.csv"
):
    """
    Creates a valid submission file for the Kaggle CUHK-X Large Model Track.

    Args:
        predictions_dict: dict mapping qa_id to the predicted string (e.g. 'A', 'BC', 'DABC')
        test_csv: path to the original test_qa.csv
        output_file: output path for the submission
    """
    df_test = pd.read_csv(test_csv)

    submission_rows = []

    for idx, row in df_test.iterrows():
        qa_id = row["qa_id"]
        category = row["category"]

        pred = predictions_dict.get(qa_id, "")

        # Fallbacks to ensure valid submission format
        pred = "".join([c for c in str(pred).upper() if c in "ABCD"])
        if not pred:
            pred = "A"  # Default fallback

        # Ensure category rules
        if category in ["single", "emotion", "object_interaction", "combination"]:
            if len(pred) > 1:
                pred = pred[0]
        elif category == "sequence":
            # Sequences must have 4 letters in some order
            if len(pred) != 4 or set(pred) != set("ABCD"):
                pred = "ABCD"  # Fallback valid sequence

        submission_rows.append({"qa_id": qa_id, "prediction": pred})

    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file} with {len(sub_df)} rows.")
    return sub_df


if __name__ == "__main__":
    # Test with dummy data
    dummy_preds = {"test_0001": "B", "test_0002": "INVALID"}
    create_submission(dummy_preds)
