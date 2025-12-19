from datasets import load_dataset


def main():
    ds = load_dataset("rjac/kaggle-entity-annotated-corpus-ner-dataset")
    ds["train"].to_csv("ner_dataset.csv", index=False)


if __name__ == "__main__":
    main()
