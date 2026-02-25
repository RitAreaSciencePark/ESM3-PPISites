import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, T5Tokenizer 

# -------------------------------
# Dataset
# -------------------------------
class ResidueInterfaceDataset(Dataset):
    def __init__(self, seqs, labels, tokenizer, max_length=1024):
        self.seqs = seqs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        lab = self.labels[idx]

        #tokenized sequence
        encoding = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        #labels
        label_tensor = torch.full((self.max_length,), -100, dtype=torch.float)
        start_idx = 1
        end_idx = min(len(lab) + start_idx, self.max_length - 1)
        label_tensor[start_idx:end_idx] = torch.tensor(lab[: end_idx - start_idx], dtype=torch.float)

        #return
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_tensor
        }