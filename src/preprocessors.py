from collections import Counter
from scipy.sparse import lil_matrix
from nltk.tokenize import wordpunct_tokenize
from numpy import array

class OneHotCustomVectorizer:
    def __init__(self, tokenizer, max_features=1024):
        self.tokenizer = tokenizer
        self.max_features = max_features
        self.vocab = {}
        
    def fit(self, sequences):
        # Tokenize and lowercase sequences
        tokenized_sequences = [self.tokenizer(seq.lower()) for seq in sequences]
        all_tokens = [token for sublist in tokenized_sequences for token in sublist]
        
        # Build vocabulary
        common_tokens = [item[0] for item in Counter(all_tokens).most_common(self.max_features)]
        self.vocab = {token: idx for idx, token in enumerate(common_tokens)}
        return self

    def transform(self, sequences):
        # Tokenize and lowercase sequences
        tokenized_sequences = [self.tokenizer(seq.lower()) for seq in sequences]
        
        # One-hot encode
        onehot_encoded = lil_matrix((len(sequences), self.max_features))
        for idx, sequence in enumerate(tokenized_sequences):
            for token in sequence:
                if token in self.vocab:
                    onehot_encoded[idx, self.vocab[token]] = 1
                    
        return onehot_encoded.tocsr()

    def fit_transform(self, sequences):
        self.fit(sequences)
        return self.transform(sequences)


class CommandTokenizer:
    def __init__(self, tokenizer_fn=wordpunct_tokenize, vocab_size=1024):
        self.tokenizer_fn = tokenizer_fn
        self.vocab_size = vocab_size
        self.token_to_int = {}
        self.UNK_TOKEN = "<UNK>"
        self.PAD_TOKEN = "<PAD>"
        
    def tokenize(self, commands):
        return [self.tokenizer_fn(cmd) for cmd in commands]

    def build_vocab(self, tokens_list):
        vocab = Counter()
        for tokens in tokens_list:
            vocab.update(tokens)
        vocab = dict(vocab.most_common(self.vocab_size - 2))  # -2 for the UNK_TOKEN and PAD_TOKEN
        self.token_to_int = {token: idx for idx, (token, _) in enumerate(vocab.items(), 2)}
        self.token_to_int[self.UNK_TOKEN] = 1 # UNK_TOKEN maps to 1
        self.token_to_int[self.PAD_TOKEN] = 0 # PAD_TOKEN maps to 0
    
    def encode(self, tokens_list):
        return [[self.token_to_int.get(token, self.token_to_int[self.UNK_TOKEN]) for token in tokens] for tokens in tokens_list]

    def pad(self, encoded_list, max_len):
        padded_list = []
        for seq in encoded_list:
            if len(seq) > max_len:
                padded_seq = seq[:max_len]
            else:
                padded_seq = seq + [0] * (max_len - len(seq))
            padded_list.append(padded_seq)
        return array(padded_list)
