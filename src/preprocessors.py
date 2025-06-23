from collections import Counter
from scipy.sparse import lil_matrix
from nltk.tokenize import wordpunct_tokenize
from numpy import array
import json
import os
from tqdm import tqdm

class OneHotCustomVectorizer:
    def __init__(self, tokenizer=wordpunct_tokenize, max_features=4096):
        self.tokenizer = tokenizer
        self.max_features = max_features
        self.vocab = {}
        self.__name__ = "OneHotCustomVectorizer"
        
    def fit(self, sequences):
        tokenized_sequences = [self.tokenizer(seq.lower()) for seq in sequences]
        all_tokens = [token for sublist in tokenized_sequences for token in sublist]
        
        common_tokens = [item[0] for item in Counter(all_tokens).most_common(self.max_features)]
        self.vocab = {token: idx for idx, token in enumerate(common_tokens)}
        return self

    def save_vocab(self, vocab_file):
        vocab_file_folder = os.path.dirname(vocab_file)
        os.makedirs(vocab_file_folder, exist_ok=True)
        with open(vocab_file, 'w') as f:
            json.dump(self.vocab, f, indent=4)
    
    def load_vocab(self, vocab_file):
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)

    def tokenize(self, sequences):
        return [self.tokenizer(seq.lower()) for seq in tqdm(sequences, desc="Tokenizing sequences")]

    def detokenize(self, idx_sequence):
        assert self.vocab != {}, "Vocabulary not built yet. Call fit() or load_vocab() first."
        return [list(self.vocab.keys())[idx] for idx in tqdm(idx_sequence, desc="Detokenizing sequences")]
    
    def transform(self, sequences):
        assert self.vocab != {}, "Vocabulary not built yet. Call fit() or load_vocab() first."
        tokenized_sequences = self.tokenize(sequences)
        
        onehot_encoded = lil_matrix((len(sequences), self.max_features))
        for idx, sequence in enumerate(tqdm(tokenized_sequences, desc="Encoding sequences")):
            for token in sequence:
                if token in self.vocab:
                    onehot_encoded[idx, self.vocab[token]] = 1
                    
        return onehot_encoded.tocsr()

    def encode(self, sequences):
        return self.transform(sequences)

    def fit_transform(self, sequences):
        self.fit(sequences)
        return self.transform(sequences)
    

class CommandTokenizer:
    def __init__(self, tokenizer_fn=wordpunct_tokenize, vocab_size=1024, max_len=256):
        self.tokenizer_fn = tokenizer_fn
        self.vocab_size = vocab_size
        self.vocab = {}
        self.UNK_TOKEN = "<UNK>"
        self.PAD_TOKEN = "<PAD>"
        self.max_len = max_len
        self.__name__ = "CommandTokenizer"
        
    def tokenize(self, commands):
        return [self.tokenizer_fn(cmd.lower()) for cmd in tqdm(commands, desc="Tokenizing commands")]
    
    def detokenize(self, idx_sequence):
        return [list(self.vocab.keys())[idx] for idx in tqdm(idx_sequence, desc="Detokenizing commands")]

    def build_vocab(self, tokens_list):
        self.vocab[self.PAD_TOKEN] = 0 # PAD_TOKEN maps to 0
        self.vocab[self.UNK_TOKEN] = 1 # UNK_TOKEN maps to 1
        vocab = Counter()
        for tokens in tokens_list:
            vocab.update(tokens)
        vocab = dict(vocab.most_common(self.vocab_size - 2))  # -2 for the UNK_TOKEN and PAD_TOKEN
        self.vocab.update({token: idx for idx, (token, _) in enumerate(vocab.items(), 2)})
    
    def dump_vocab(self, vocab_file):
        vocab_file_folder = os.path.dirname(vocab_file)
        os.makedirs(vocab_file_folder, exist_ok=True)
        with open(vocab_file, 'w') as f:
            json.dump(self.vocab, f, indent=4)
    
    def load_vocab(self, vocab_file):
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
    
    def encode(self, tokens_list):
        assert self.vocab != {}, "Vocabulary not built yet. Call build_vocab() first."
        return [[self.vocab.get(token, self.vocab[self.UNK_TOKEN]) for token in tokens] for tokens in tokens_list]

    def pad(self, encoded_list):
        padded_list = []
        for seq in encoded_list:
            if len(seq) > self.max_len:
                padded_seq = seq[:self.max_len]
            else:
                padded_seq = seq + [0] * (self.max_len - len(seq))
            padded_list.append(padded_seq)
        return array(padded_list)

    def transform(self, commands):
        tokenized_commands = self.tokenize(commands)
        encoded_commands = self.encode(tokenized_commands)
        padded_commands = self.pad(encoded_commands)
        return padded_commands
