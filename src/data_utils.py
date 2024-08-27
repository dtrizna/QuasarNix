import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from scipy.sparse import csr_matrix

from .preprocessors import CommandTokenizer, OneHotCustomVectorizer
from typing import List, Union, Callable

import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from nltk.tokenize import wordpunct_tokenize


class CSRTensorDataset(Dataset):
    def __init__(self, csr_data, labels=None):
        if labels is not None:
            assert csr_data.shape[0] == len(labels)
        self.csr_data = csr_data
        self.labels = labels

    def get_tensor(self, row):
        assert isinstance(row, (np.ndarray, torch.Tensor)),\
            "Expected row to be a numpy array or torch tensor, but got {}".format(type(row))
        if isinstance(row, np.ndarray):
            data = torch.from_numpy(row).float()
        elif isinstance(row, torch.Tensor):
            data = row.clone().detach().float()
        return data

    def __len__(self):
        return self.csr_data.shape[0]

    def __getitem__(self, index):
        row = self.csr_data[index].toarray().squeeze()  # Convert the sparse row to a dense numpy array
        x_data = self.get_tensor(row)
        if self.labels is not None:
            y_data = self.get_tensor(self.labels[index])
            return x_data, y_data
        else:
            return x_data,


def create_dataloader(X, y=None, batch_size=1024, shuffle=False, workers=4):
    # Convert numpy arrays to torch tensors
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).long()
    if y is not None and isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()

    # Handle csr_matrix case
    if isinstance(X, csr_matrix):
        dataset = CSRTensorDataset(X, y)
    # Handle torch.Tensor case
    elif isinstance(X, torch.Tensor):
        dataset = TensorDataset(X, y) if y is not None else TensorDataset(X)
    else:
        raise ValueError("Unsupported type for X. Supported types are numpy arrays, torch tensors, and scipy CSR matrices.")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, persistent_workers=True, pin_memory=True)


def commands_to_loader(
        cmd: List[str], 
        tokenizer: Union[CommandTokenizer, OneHotCustomVectorizer], 
        workers: int, 
        batch_size: int, 
        y: np.ndarray = None,
        shuffle: bool = False
) -> DataLoader:
    """Convert a list of commands to a DataLoader."""
    array = tokenizer.transform(cmd)
    if y is None:
        loader = create_dataloader(array, batch_size=batch_size, workers=workers, shuffle=shuffle)
    else:
        loader = create_dataloader(array, y, batch_size=batch_size, workers=workers, shuffle=shuffle)
    return loader


def load_data(root, seed=33, limit=None):
    """
    NOTE: 
        First shuffle the data -- to take random elements from each class.
        limit//2 -- since there are 2 classes, so full data size is limit.
        Second shuffle the data -- to mix the two classes.
    """
    train_base_parquet_file = [x for x in os.listdir(os.path.join(root,'data/train_baseline.parquet/')) if x.endswith('.parquet')][0]
    test_base_parquet_file = [x for x in os.listdir(os.path.join(root,'data/test_baseline.parquet/')) if x.endswith('.parquet')][0]
    train_rvrs_parquet_file = [x for x in os.listdir(os.path.join(root,'data/train_rvrs.parquet/')) if x.endswith('.parquet')][0]
    test_rvrs_parquet_file = [x for x in os.listdir(os.path.join(root,'data/test_rvrs.parquet/')) if x.endswith('.parquet')][0]

    train_baseline_df = pd.read_parquet(os.path.join(root,'data/train_baseline.parquet/', train_base_parquet_file))
    test_baseline_df = pd.read_parquet(os.path.join(root,'data/test_baseline.parquet/', test_base_parquet_file))
    train_malicious_df = pd.read_parquet(os.path.join(root,'data/train_rvrs.parquet/', train_rvrs_parquet_file))
    test_malicious_df = pd.read_parquet(os.path.join(root,'data/test_rvrs.parquet/', test_rvrs_parquet_file))

    if limit is not None:
        X_train_baseline_cmd = shuffle(train_baseline_df['cmd'].values.tolist(), random_state=seed)[:limit//2]
        X_train_malicious_cmd = shuffle(train_malicious_df['cmd'].values.tolist(), random_state=seed)[:limit//2]
        X_test_baseline_cmd = shuffle(test_baseline_df['cmd'].values.tolist(), random_state=seed)[:limit//2]
        X_test_malicious_cmd = shuffle(test_malicious_df['cmd'].values.tolist(), random_state=seed)[:limit//2]
    else:
        X_train_baseline_cmd = train_baseline_df['cmd'].values.tolist()
        X_train_malicious_cmd = train_malicious_df['cmd'].values.tolist()
        X_test_baseline_cmd = test_baseline_df['cmd'].values.tolist()
        X_test_malicious_cmd = test_malicious_df['cmd'].values.tolist()

    X_train_non_shuffled = X_train_baseline_cmd + X_train_malicious_cmd
    y_train = np.array([0] * len(X_train_baseline_cmd) + [1] * len(X_train_malicious_cmd), dtype=np.int8)
    X_train_cmds, y_train = shuffle(X_train_non_shuffled, y_train, random_state=seed)

    X_test_non_shuffled = X_test_baseline_cmd + X_test_malicious_cmd
    y_test = np.array([0] * len(X_test_baseline_cmd) + [1] * len(X_test_malicious_cmd), dtype=np.int8)
    X_test_cmds, y_test = shuffle(X_test_non_shuffled, y_test, random_state=seed)

    return X_train_cmds, y_train, X_test_cmds, y_test, X_train_malicious_cmd, X_train_baseline_cmd, X_test_malicious_cmd, X_test_baseline_cmd


def load_nl2bash(root):
    with open(os.path.join(root, "data", "nl2bash.cm"), "r", encoding="utf-8") as f:
        baseline = f.readlines()
    return baseline


def load_tokenizer(
        tokenizer_type: str,
        train_cmds: List[str],
        vocab_size: int,
        max_len: int,
        tokenizer_fn: Callable = wordpunct_tokenize,
        suffix: str = "",
        logs_folder: str = "./"
) -> Union[OneHotCustomVectorizer, CommandTokenizer]:
    if "onehot" in tokenizer_type:
        oh_file = os.path.join(logs_folder, f"onehot_vocab_{vocab_size}{suffix}.pkl")
        if os.path.exists(oh_file):
            print(f"[!] Loading One-Hot encoder from '{oh_file}'...")
            with open(oh_file, "rb") as f:
                tokenizer = pickle.load(f)
        else:
            tokenizer = OneHotCustomVectorizer(tokenizer=tokenizer_fn, max_features=vocab_size)
            print("[*] Fitting One-Hot encoder...")
            now = time.time()
            tokenizer.fit(train_cmds)
            print(f"[!] Fitting One-Hot encoder took: {time.time() - now:.2f}s") # ~90s
            with open(oh_file, "wb") as f:
                pickle.dump(tokenizer, f)
    else:
        tokenizer = CommandTokenizer(tokenizer_fn=tokenizer_fn, vocab_size=vocab_size, max_len=max_len)
        vocab_file = os.path.join(logs_folder, f"wordpunct_vocab_{vocab_size}{suffix}.json")
        if os.path.exists(vocab_file):
                print(f"[!] Loading vocab from '{vocab_file}'...")
                tokenizer.load_vocab(vocab_file)
        else:
            print("[*] Building Tokenizer for Embedding vocab and encoding...")
            X_train_tokens_poisoned = tokenizer.tokenize(train_cmds)
            tokenizer.build_vocab(X_train_tokens_poisoned)
            tokenizer.dump_vocab(vocab_file)
    
    return tokenizer
