from datasets import load_dataset
import pandas as pd


def load_arxiv_dataset():
    """
    Downloads the ML-ArXiv dataset from Hugging Face.
    """
    print("Downloading dataset...")

    dataset = load_dataset("CShorten/ML-ArXiv-Papers")

    print("Dataset downloaded successfully!")

    return dataset["train"]


def preprocess_data(dataset):
    """
    Converts the dataset to a pandas DataFrame and cleans it.
    """

    print("Converting dataset to DataFrame...")

    df = dataset.to_pandas()

    print(f"Original Shape: {df.shape}")

    # Keep only required columns
    df = df[["title", "abstract"]]

    # Remove missing values
    df.dropna(inplace=True)

    # Remove duplicate papers
    df.drop_duplicates(inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    print(f"\nCleaned Shape: {df.shape}")

    print("\nFirst 5 rows:")
    print(df.head())

    # Save cleaned dataset
    df.to_csv("data/ml_arxiv_papers.csv", index=False)

    print("\n✅ Cleaned dataset saved to data/ml_arxiv_papers.csv")

    return df
def main():

    dataset = load_arxiv_dataset()

    df = preprocess_data(dataset)

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()