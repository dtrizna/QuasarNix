import os
import sentencepiece as spm

# build BPE tokenizer based on sentencepiece with similar API like sklearn
class BPEVectorizer:
    def __init__(self, vocab_size=10000, model_prefix='bpe', model_type='unigram', vocab_file=None):
        self.vocab_size = vocab_size
        self.model_prefix = model_prefix
        self.model_type = model_type
        self.vocab_file = vocab_file
        self.tokenizer = None
        self.vocab = None

    def fit(self, texts):
        with open('tmp.txt', 'w') as f:
            for text in texts:
                f.write(text + '\n')
        spm_command = '--input=tmp.txt --model_prefix={} --vocab_size={} --model_type={}'.format(
                self.model_prefix, self.vocab_size, self.model_type)
        print(f'[*] Training sentencepiece model: {spm_command}')
        spm.SentencePieceTrainer.Train(spm_command)
        os.remove('tmp.txt')
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load('{}.model'.format(self.model_prefix))
        self.vocab = {self.tokenizer.IdToPiece(i): i for i in range(self.tokenizer.GetPieceSize())}
        if self.vocab_file is not None:
            with open(self.vocab_file, 'w') as f:
                for i in range(self.tokenizer.GetPieceSize()):
                    f.write(self.tokenizer.IdToPiece(i) + '\n')

    def transform(self, texts):
        assert self.tokenizer is not None
        return [self.tokenizer.EncodeAsIds(text) for text in texts]

    def tokenize(self, texts):
        assert self.tokenizer is not None
        if isinstance(texts, str):
            texts = [texts]
        return [self.tokenizer.EncodeAsPieces(text) for text in texts]

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

    def save(self, model_path):
        assert self.tokenizer is not None
        self.tokenizer.Save(model_path)

    def load(self, model_path):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(model_path)
        self.vocab = {self.tokenizer.IdToPiece(i): i for i in range(self.tokenizer.GetPieceSize())}

    def encode(self, text):
        assert self.tokenizer is not None
        return self.tokenizer.EncodeAsIds(text)

    def decode(self, ids):
        assert self.tokenizer is not None
        return self.tokenizer.DecodeIds(ids)

    def __len__(self):
        return self.tokenizer.GetPieceSize()

    def __getitem__(self, item):
        return self.tokenizer.IdToPiece(item)

    def __contains__(self, item):
        return item in self.vocab
