from torch.utils.data.dataset import Dataset
import torch


def digits2position(vec_of_digits, n_positions):
    """One-hot encoding of a batch of vectors."""
    return torch.eye(n_positions)[vec_of_digits]


class OneHotDataset(Dataset):
    def __init__(self, data, labels, n_classes):
        self.data = data
        self.labels = digits2position(labels, n_classes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
