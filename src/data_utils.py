import os
import re
import time
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from nltk.tokenize import wordpunct_tokenize
from typing import List, Union, Callable

from src.preprocessors import CommandTokenizer, OneHotCustomVectorizer
from src.augmentation import NixCommandAugmentationWithBaseline, REVERSE_SHELL_TEMPLATES

import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from scipy.sparse import csr_matrix


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
        y: np.ndarray | None = None,
        shuffle: bool = False
) -> DataLoader:
    """Convert a list of commands to a DataLoader."""
    array = tokenizer.transform(cmd)
    if y is None:
        loader = create_dataloader(array, batch_size=batch_size, workers=workers, shuffle=shuffle)
    else:
        loader = create_dataloader(array, y, batch_size=batch_size, workers=workers, shuffle=shuffle)
    return loader


def generate_synthetic_data(paths: dict, root: Path, seed: int = 33):
    baseline = load_nl2bash(root)
    synthetic_data_generator = NixCommandAugmentationWithBaseline(
        templates=REVERSE_SHELL_TEMPLATES,
        legitimate_baseline=baseline,
        random_state=seed
    )

    number_of_examples_per_template = len(baseline) // len(REVERSE_SHELL_TEMPLATES)
    synthetic_malicious = synthetic_data_generator.generate_commands(number_of_examples_per_template)

    # train/test split
    train_malicious, test_malicious = train_test_split(synthetic_malicious, test_size=0.2, random_state=seed)
    train_baseline, test_baseline = train_test_split(baseline, test_size=0.2, random_state=seed)

    # transform to dataframe
    train_malicious_df = pd.DataFrame({'cmd': train_malicious})
    test_malicious_df = pd.DataFrame({'cmd': test_malicious})
    train_baseline_df = pd.DataFrame({'cmd': train_baseline})
    test_baseline_df = pd.DataFrame({'cmd': test_baseline})

    # save to parquet
    train_malicious_df.to_parquet(paths['train_malicious'])
    test_malicious_df.to_parquet(paths['test_malicious'])
    train_baseline_df.to_parquet(paths['train_baseline'])
    test_baseline_df.to_parquet(paths['test_baseline'])

    print(f"[+] Synthetic data generated and saved to:\n{json.dumps({k: str(v) for k,v in paths.items()}, indent=4)}")
    return


def load_data(seed: int = 33, limit: int = None, root: Path | None = None):
    """
    Load and prepare training and testing data from parquet files.
    
    Args:
        root: Root directory containing the data folders
        seed: Random seed for shuffling
        limit: If set, limits the number of samples per class
        
    Returns:
        Tuple of (train_cmds, train_labels, test_cmds, test_labels,
                 train_malicious, train_baseline, test_malicious, test_baseline)
    """
    if root is None:
        root = Path(__file__).parent.parent

    data_root = Path(root) / 'data' / 'nix_shell'
    paths = {
        'train_baseline': data_root / 'train_baseline.parquet',
        'test_baseline': data_root / 'test_baseline.parquet',
        'train_malicious': data_root / 'train_rvrs.parquet',
        'test_malicious': data_root / 'test_rvrs.parquet'
    }

    # if do not exist
    exist = all(path.exists() for path in paths.values())
    if not exist:
        print(f"[-] Data files do not exist in {data_root}")
        print(f"    Do you want to generate synthetic data from NL2Bash dataset? (Y/n)")
        if input().lower() in ["y", "yes", ""]:
            generate_synthetic_data(paths, root, seed)
        else:
            raise FileNotFoundError(f"Data files do not exist in {data_root}")
    
    def _load_parquet(path: Path):
        # might be directly parquet file,
        # might be a folder with parquet file inside
        if path.is_dir():
            return pd.read_parquet(next(path.glob('*.parquet')))
        else:
            return pd.read_parquet(path)

    # Load dataframes
    dfs = {
        name: _load_parquet(path)
        for name, path in paths.items()
    }

    # Extract commands with optional limit
    def get_commands(df, size=None):
        cmds = df['cmd'].values.tolist()
        if size is not None:
            cmds = shuffle(cmds, random_state=seed)[:size]
        return cmds

    size_per_class = limit // 2 if limit else None
    
    X_train_baseline = get_commands(dfs['train_baseline'], size_per_class)
    X_train_malicious = get_commands(dfs['train_malicious'], size_per_class)
    X_test_baseline = get_commands(dfs['test_baseline'], size_per_class)
    X_test_malicious = get_commands(dfs['test_malicious'], size_per_class)

    # Combine and shuffle training data
    X_train = X_train_baseline + X_train_malicious
    y_train = np.array([0] * len(X_train_baseline) + [1] * len(X_train_malicious), dtype=np.int8)
    X_train, y_train = shuffle(X_train, y_train, random_state=seed)

    # Combine and shuffle test data
    X_test = X_test_baseline + X_test_malicious
    y_test = np.array([0] * len(X_test_baseline) + [1] * len(X_test_malicious), dtype=np.int8)
    X_test, y_test = shuffle(X_test, y_test, random_state=seed)

    return X_train, y_train, X_test, y_test, X_train_malicious, X_train_baseline, X_test_malicious, X_test_baseline

def load_nl2bash(root):
    with open(Path(root) / "data" / "nix_shell" / "nl2bash.cm", "r", encoding="utf-8") as f:
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

# powershell collections
def read_powershell_pan_enc(root: str):
    file = os.path.join(root, "data", "dtrizna", "powershell", "pan42", "ps_encodedcommand_data_clean.txt")
    with open(file, "r") as f:
        lines = f.readlines()
    return lines


def read_powershell_offensive(root: str):
    folder = os.path.join(root, "data", "dtrizna", "powershell", "offensive-powershell", "data")
    for parquet_file in os.listdir(folder):
        if parquet_file.endswith(".parquet"):
            if "train" in parquet_file:
                train_df = pd.read_parquet(os.path.join(folder, parquet_file))
            elif "dev" in parquet_file:
                dev_df = pd.read_parquet(os.path.join(folder, parquet_file))
            elif "test" in parquet_file:
                test_df = pd.read_parquet(os.path.join(folder, parquet_file))
    all_pwsh = np.concatenate([train_df['code'].values, dev_df['code'].values, test_df['code'].values]).tolist()
    return all_pwsh


def convert_powershell_ps1_to_oneliner(script: str) -> str:
    lines = script.splitlines()
    lines = [line for line in lines if not line.strip().startswith("#")]
    lines = [line for line in lines if line.strip()]
    oneliner = ' '.join('; '.join(lines).split())
    # remove param block using regex
    oneliner = re.sub(r"param\s*\([^)]*\)", "", oneliner)
    return oneliner.lstrip("; .")

def read_powershell_corpus(root: str, limit: int = None):
    dataset = []     
    
    corpus_folder = os.path.join(root, "data", "dtrizna", "powershell", "powershell_collection")
    if not os.path.exists(corpus_folder):
        raise FileNotFoundError(f"Corpus folder '{corpus_folder}' does not exist.")
    
    l = limit if limit is not None else None
    for dirpath, _, filenames in tqdm(os.walk(corpus_folder), desc="[*] Reading PowerShell corpus", total=l):
        for filename in filenames:
            if filename.endswith(".ps1") or filename.endswith(".psm1"):
                filepath = os.path.join(dirpath, filename)
                # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    script = f.read()
                oneliner = convert_powershell_ps1_to_oneliner(script)
                dataset.append(oneliner)

                if limit is not None and len(dataset) >= limit:
                    return dataset
                
    return dataset

def read_powershell_data(root: str, seed: int = 33, split_train_test: bool = True, limit: int = None):
    pan_pwsh = read_powershell_pan_enc(root)
    huggingface_pwsh = read_powershell_offensive(root)
    powershell_collection_pwsh = read_powershell_corpus(root, limit=limit)

    malicious = pan_pwsh + huggingface_pwsh
    benign = powershell_collection_pwsh

    malicious_labels = np.array([1] * len(malicious), dtype=np.int8)
    benign_labels = np.array([0] * len(benign), dtype=np.int8)

    X = malicious + benign
    y = np.concatenate([malicious_labels, benign_labels])
    X, y = shuffle(X, y, random_state=seed)

    if limit is not None:
        X, y = X[:limit], y[:limit]

    if split_train_test:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        return X_train, X_test, y_train, y_test
    else:
        return X, y

def read_powershell_parquet(root: str, seed: int = 33, limit: int = None):
    train_df = pd.read_parquet(os.path.join(root, "data", "powershell", f"train_pwsh_seed_{seed}.parquet"))
    test_df = pd.read_parquet(os.path.join(root, "data", "powershell", f"test_pwsh_seed_{seed}.parquet"))
    
    X_train = train_df['code'].values.tolist()
    y_train = train_df['label'].values

    X_test = test_df['code'].values.tolist()
    y_test = test_df['label'].values

    if limit is not None:
        X_train, y_train = X_train[:limit], y_train[:limit]
        X_test, y_test = X_test[:limit], y_test[:limit]

    return X_train, X_test, y_train, y_test

