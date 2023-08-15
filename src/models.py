import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torchmetrics
import lightning as L
import math

from sklearn.metrics import roc_curve

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, dropout=None):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Add dropout layer if dropout probability is provided
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        
        # Apply dropout if the layer is available
        if self.dropout:
            x = self.dropout(x)
        
        x = self.fc2(x)
        return x


class SimpleMLPWithEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, hidden_dim=32, use_positional_encoding=False, max_seq_len=500, device=None, dropout=None):
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
