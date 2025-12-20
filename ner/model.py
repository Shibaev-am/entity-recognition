import pytorch_lightning as pl
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.optim import AdamW
from transformers import BertForTokenClassification


class BERTNERModel(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr, idx2tag):
        super().__init__()
        self.save_hyperparameters()
        self.model = BertForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.lr = lr
        self.idx2tag = idx2tag
        self.validation_step_outputs = []
        self.training_step_outputs = []

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        loss = outputs.loss
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        logits = outputs.logits
        preds = torch.argmax(logits, dim=2)
        self.training_step_outputs.append(
            {"preds": preds.cpu(), "labels": batch["labels"].cpu()}
        )
        return loss

    def on_train_epoch_end(self):
        preds_list = []
        targets_list = []

        for output in self.training_step_outputs:
            for i in range(len(output["preds"])):
                temp_p = []
                temp_t = []
                p_row = output["preds"][i]
                t_row = output["labels"][i]

                for j in range(len(t_row)):
                    if t_row[j] != -100:
                        temp_p.append(self.idx2tag[p_row[j].item()])
                        temp_t.append(self.idx2tag[t_row[j].item()])

                preds_list.append(temp_p)
                targets_list.append(temp_t)

        train_f1 = f1_score(targets_list, preds_list)
        self.log("train_f1", train_f1, prog_bar=True, logger=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        loss = outputs.loss
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        logits = outputs.logits
        preds = torch.argmax(logits, dim=2)

        self.validation_step_outputs.append(
            {"preds": preds.cpu(), "labels": batch["labels"].cpu()}
        )
        return loss

    def on_validation_epoch_end(self):
        preds_list = []
        targets_list = []

        for output in self.validation_step_outputs:
            for i in range(len(output["preds"])):
                temp_p = []
                temp_t = []
                p_row = output["preds"][i]
                t_row = output["labels"][i]

                for j in range(len(t_row)):
                    if t_row[j] != -100:
                        temp_p.append(self.idx2tag[p_row[j].item()])
                        temp_t.append(self.idx2tag[t_row[j].item()])

                preds_list.append(temp_p)
                targets_list.append(temp_t)

        f1 = f1_score(targets_list, preds_list)
        precision = precision_score(targets_list, preds_list)
        recall = recall_score(targets_list, preds_list)

        self.log("val_f1", f1, prog_bar=True, logger=True)
        self.log("val_precision", precision, prog_bar=False, logger=True)
        self.log("val_recall", recall, prog_bar=False, logger=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)
