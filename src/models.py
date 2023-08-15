import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torchmetrics
import lightning as L
import math

from sklearn.metrics import roc_curve

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[32], dropout=None):
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]
        
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        # Dynamically create hidden layers based on hidden_dim
        for h_dim in hidden_dim:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class SimpleMLPWithEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, hidden_dim=[32], use_positional_encoding=False, max_seq_len=500, device=None, dropout=None):
        super(SimpleMLPWithEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.mlp = SimpleMLP(input_dim=embedding_dim, output_dim=output_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.use_positional_encoding = use_positional_encoding
        
        # Positional encoding
        if self.use_positional_encoding:
            self.positional_encoding = self.init_positional_encoding(embedding_dim, max_seq_len)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def init_positional_encoding(self, d_model, max_seq_len):
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        x = self.embedding(x)
        if self.use_positional_encoding:
            x = x + self.positional_encoding[:, :x.size(1), :].to(self.device)
        x = x.mean(dim=1)
        return self.mlp(x)


class MLP_LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.fpr = 1e-4

        self.loss = BCEWithLogitsLoss()

        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.train_f1 = torchmetrics.F1Score(task='binary', average='macro')
        self.train_auc = torchmetrics.AUROC(task='binary')
        self.train_tpr = self.get_tpr_at_fpr

        self.val_acc = torchmetrics.Accuracy(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary', average='macro')
        self.val_auc = torchmetrics.AUROC(task='binary')
        self.val_tpr = self.get_tpr_at_fpr

        self.test_acc = torchmetrics.Accuracy(task='binary')
        self.test_f1 = torchmetrics.F1Score(task='binary', average='macro')
        self.test_auc = torchmetrics.AUROC(task='binary')
        self.test_tpr = self.get_tpr_at_fpr

        self.save_hyperparameters(ignore=["model"])
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_tpr_at_fpr(self, predicted_logits, true_labels, fprNeeded=1e-4):
        predicted_probs = torch.sigmoid(predicted_logits).cpu().detach().numpy()
        true_labels = true_labels.cpu().detach().numpy()
        fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
        if all(np.isnan(fpr)):
            return np.nan#, np.nan
        else:
            tpr_at_fpr = tpr[fpr <= fprNeeded][-1]
            #threshold_at_fpr = thresholds[fpr <= fprNeeded][-1]
            return tpr_at_fpr#, threshold_at_fpr

    def forward(self, x):
        return self.model(x)
    
    def _shared_step(self, batch):
        # used by training_step, validation_step and test_step
        x, y = batch
        y = y.unsqueeze(1)
        logits = self(x)
        loss = self.loss(logits, y)
        
        return loss, y, logits
    
    def training_step(self, batch, batch_idx):
        loss, y, logits = self._shared_step(batch)
        self.log('train_loss', loss, prog_bar=True) # logging loss for this mini-batch
        self.train_acc(logits, y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.train_f1(logits, y)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.train_auc(logits, y)
        self.log('train_auc', self.train_auc, on_step=False, on_epoch=True, prog_bar=True)
        train_tpr = self.train_tpr(logits, y, fprNeeded=self.fpr)
        self.log('train_tpr', train_tpr, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y, logits = self._shared_step(batch)
        self.log('val_loss', loss)
        self.val_acc(logits, y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.val_f1(logits, y)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.val_auc(logits, y)
        self.log('val_auc', self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        val_tpr = self.val_tpr(logits, y, fprNeeded=self.fpr)
        self.log('val_tpr', val_tpr, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, y, logits = self._shared_step(batch)
        self.log('test_loss', loss)
        self.test_acc(logits, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.test_f1(logits, y)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.test_auc(logits, y)
        self.log('test_auc', self.test_auc, on_step=False, on_epoch=True, prog_bar=True)
        test_tpr = self.test_tpr(logits, y, fprNeeded=self.fpr)
        self.log('test_tpr', test_tpr, on_step=False, on_epoch=True, prog_bar=True)
        return loss


class CNN1DGroupedModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_channels, kernel_sizes, mlp_hidden_dims, output_dim, dropout=None):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.grouped_convs = nn.ModuleList([nn.Conv1d(embed_dim, num_channels, kernel) for kernel in kernel_sizes])
        
        mlp_input_dim = num_channels * len(kernel_sizes)
        self.mlp = SimpleMLP(input_dim=mlp_input_dim, output_dim=output_dim, hidden_dim=mlp_hidden_dims, dropout=dropout)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        conv_outputs = [conv(x) for conv in self.grouped_convs]
        x = torch.cat(conv_outputs, dim=1)
        x = x.view(x.size(0), -1)  # Flatten
        return self.mlp(x)


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, mlp_hidden_dims, output_dim, lstm_layers=2, dropout=None):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=lstm_layers, bidirectional=True, batch_first=True, dropout=(dropout if dropout else 0))
        
        mlp_input_dim = 2 * hidden_dim  # bidirectional -> concatenate forward & backward
        self.mlp = SimpleMLP(input_dim=mlp_input_dim, output_dim=output_dim, hidden_dim=mlp_hidden_dims, dropout=dropout)

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        x = torch.cat((hn[0], hn[1]), dim=1)  # Concatenate the final forward and backward hidden layers
        return self.mlp(x)


class CNN1D_BiLSTM_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_channels, kernel_size, lstm_hidden_dim, mlp_hidden_dims, output_dim, lstm_layers=2, dropout=None):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1d = nn.Conv1d(embed_dim, num_channels, kernel_size)
        self.lstm = nn.LSTM(num_channels, lstm_hidden_dim, num_layers=lstm_layers, bidirectional=True, batch_first=True, dropout=(dropout if dropout else 0))
        
        mlp_input_dim = 2 * lstm_hidden_dim  # bidirectional -> concatenate forward & backward
        self.mlp = SimpleMLP(input_dim=mlp_input_dim, output_dim=output_dim, hidden_dim=mlp_hidden_dims, dropout=dropout)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.conv1d(x).transpose(1, 2)
        _, (hn, _) = self.lstm(x)
        x = torch.cat((hn[0], hn[1]), dim=1)  # Concatenate the final forward and backward hidden layers
        return self.mlp(x)


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, mlp_hidden_dims, max_len, dropout=None, device=None, output_dim=1):
        super(TransformerEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = self.init_positional_encoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = SimpleMLP(input_dim=d_model, output_dim=output_dim, hidden_dim=mlp_hidden_dims, dropout=dropout)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
    def init_positional_encoding(self, d_model, max_seq_len):
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = src + self.pos_encoder(src).to(self.device)
        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return self.decoder(output)
