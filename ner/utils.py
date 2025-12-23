import shutil
from pathlib import Path

import pandas as pd


def download_data(data_dir: str, filename: str):
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    target_path = data_path / filename

    if target_path.exists():
        print(f"Data check: File found at {target_path}")
        return

    possible_paths = [
        Path(f"/kaggle/input/entity-annotated-corpus/{filename}"),
        Path(filename),
    ]

    for path in possible_paths:
        if path.exists():
            print(f"Copying data from {path} to {target_path}")
            shutil.copy(path, target_path)
            return

    raise FileNotFoundError(
        f"Could not find {filename} in {data_dir}. "
        f"Please run 'wget -O {target_path} <url>' to download it."
    )


def load_and_clean_data(file_path: str):
    print(f"Loading data from: {file_path}")

    try:
        data = pd.read_csv(file_path, encoding="latin1")
        if len(data.columns) < 2:
            print("Warning: CSV format looks wrong, trying delimiter=';'")
            data = pd.read_csv(file_path, encoding="latin1", sep=";")
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}")

    data.columns = data.columns.str.strip()
    print(f"Original columns: {data.columns.tolist()}")

    rename_map = {
        "sentence_idx": "Sentence #",
        "Sentence": "Sentence #",
        "sentence": "Sentence #",
        "Sentence ID": "Sentence #",
        "word": "Word",
        "tag": "Tag",
        "Tag": "Tag",
        "sentence_id": "Sentence #",
        "tokens": "Word",
        "ner_tags": "Tag",
    }

    data.rename(columns=rename_map, inplace=True)
    print(f"Renamed columns: {data.columns.tolist()}")

    if "Sentence #" not in data.columns:
        raise KeyError(
            f"Column 'Sentence #' not found after renaming! Current columns: {data.columns.tolist()}"
        )

    data = data.ffill()

    data["Sentence #"] = data["Sentence #"].astype(str)

    print("Data loaded and cleaned successfully.")
    return data
