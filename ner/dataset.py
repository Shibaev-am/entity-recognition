from pathlib import Path

import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast

from ner.utils import download_data, load_and_clean_data


class BertNERDataset(Dataset):
    def __init__(self, sentences, tags, tag2idx, tokenizer, max_len):
        self.sentences = sentences
        self.tags = tags
        self.tag2idx = tag2idx
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        word_labels = self.tags[idx]

        encoding = self.tokenizer(
            sentence,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )

        labels = []
        word_ids = encoding.word_ids()
        prev_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != prev_word_idx:
                labels.append(self.tag2idx[word_labels[word_idx]])
            else:
                labels.append(-100)
            prev_word_idx = word_idx

        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item["labels"] = torch.as_tensor(labels)
        if "offset_mapping" in item:
            item.pop("offset_mapping")
        return item


class NERDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = BertTokenizerFast.from_pretrained(cfg.model.name)
        self.tag2idx = {}
        self.idx2tag = {}

    def prepare_data(self):
        download_data(self.cfg.paths.data_dir, self.cfg.data.csv_filename)

    def setup(self, stage=None):
        file_path = Path(self.cfg.paths.data_dir) / self.cfg.data.csv_filename
        data = load_and_clean_data(file_path)

        sentences = data.groupby("Sentence #")["Word"].apply(list).values
        tags = data.groupby("Sentence #")["Tag"].apply(list).values

        tag_vals = list(set(data["Tag"].values))
        self.tag2idx = {t: i for i, t in enumerate(tag_vals)}
        self.idx2tag = {i: t for t, i in self.tag2idx.items()}

        train_s, val_s, train_t, val_t = train_test_split(
            sentences,
            tags,
            test_size=self.cfg.data.test_size,
            random_state=self.cfg.seed,
        )

        self.train_ds = BertNERDataset(
            train_s, train_t, self.tag2idx, self.tokenizer, self.cfg.data.max_length
        )
        self.val_ds = BertNERDataset(
            val_s, val_t, self.tag2idx, self.tokenizer, self.cfg.data.max_length
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
        )
