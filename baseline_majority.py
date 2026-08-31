import pandas as pd


def run_majority_baseline():
    print("Loading data...")
    train = pd.read_csv("training_qa.csv")
    test = pd.read_csv("test_qa.csv")

    print("Computing category-specific majority classes...")
    # Find the most frequent answer per category
    majority_classes = (
        train.groupby("category")["answer"].apply(lambda x: x.mode()[0]).to_dict()
    )

    print(f"Majority Classes per Category: {majority_classes}")

    # Predict on test set
    predictions = test["category"].map(majority_classes)

    # Fallback to absolute majority if a category is missing
    absolute_majority = train["answer"].mode()[0]
    predictions.fillna(absolute_majority, inplace=True)

    # Save submission
    test["prediction"] = predictions
    submission = test[["qa_id", "prediction"]]
    submission.to_csv("submission_majority.csv", index=False)
    print("Saved submission_majority.csv")


if __name__ == "__main__":
    run_majority_baseline()
