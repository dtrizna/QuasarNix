from torch.utils.data import TensorDataset, Dataset, DataLoader

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse import csr_matrix


class CSRTensorDataset(Dataset):
    def __init__(self, csr_data, labels):
        assert csr_data.shape[0] == len(labels)
        self.csr_data = csr_data
        self.labels = labels

    def __len__(self):
        return self.csr_data.shape[0]

    def __getitem__(self, index):
        row = self.csr_data[index].toarray().squeeze()  # Convert the sparse row to a dense numpy array
        label = self.labels[index]
        return torch.tensor(row, dtype=torch.long), torch.tensor(label, dtype=torch.float32)


def create_dataloader(X, y, batch_size, shuffle=False, workers=4):
    if isinstance(X, csr_matrix):
        dataset = CSRTensorDataset(X, y)    
    elif isinstance(X, np.ndarray):
        X = torch.from_numpy(X).long()
        y = torch.from_numpy(y).float()
        dataset = TensorDataset(X, y)    
    elif isinstance(X, torch.Tensor):
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y).float()
        dataset = TensorDataset(X, y)    
    else:
        raise ValueError("Unsupported type for X. Supported types are numpy arrays, torch tensors, and scipy CSR matrices.")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, persistent_workers=True, pin_memory=True)
