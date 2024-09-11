from typing import Any, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss

import torchmetrics
import lightning as L
import math

from sklearn.metrics import roc_curve


class PyTorchLightningModel(L.LightningModule):
    """
    NOTE: Deprecated, use .lit_utils.PyTorchLightningModel instead.
    """
    def __init__(
            self,
            model: nn.Module,
            learning_rate: float,
            fpr: float = 1e-4,
            scheduler: Union[None, str] = None,
            scheduler_step_budget: Union[None, int] = None
    ):
        # NOTE: scheduler_step_budget = epochs * len(train_loader)
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.fpr = fpr
        self.loss = BCEWithLogitsLoss()

        assert scheduler in [None, "onecycle", "cosine"], "Scheduler must be onecycle or cosine"
        if scheduler is not None:
            assert isinstance(scheduler_step_budget, int), "Scheduler step budget must be provided"
            print(f"[!] Scheduler: {scheduler} | Scheduler step budget: {scheduler_step_budget}")
        self.scheduler = scheduler
        self.scheduler_step_budget = scheduler_step_budget

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        if self.scheduler is None:
            return optimizer

        if self.scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.scheduler_step_budget
            )
        if self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_step_budget
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step", # default: epoch
                "frequency": 1
            }
        }


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
        # NOTE: keep batch_idx -- lightning needs it
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
        learning_rate = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', learning_rate, on_step=False, on_epoch=True, prog_bar=False)
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

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return super().predict_step(batch[0], batch_idx, dataloader_idx)
        


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
    def __init__(self, vocab_size, embedding_dim, output_dim, hidden_dim=[32], use_positional_encoding=False, max_len=500, dropout=None):
        super(SimpleMLPWithEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.mlp = SimpleMLP(input_dim=embedding_dim, output_dim=output_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.use_positional_encoding = use_positional_encoding
        
        # Positional encoding
        if self.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(embedding_dim, dropout, max_len)

    def forward(self, x):
        x = self.embedding(x)
        if self.use_positional_encoding:
            x = self.positional_encoding(x)
        x = x.mean(dim=1)
        return self.mlp(x)


class CNN1DGroupedModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_channels, kernel_sizes, mlp_hidden_dims, output_dim, dropout=None):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.grouped_convs = nn.ModuleList([nn.Conv1d(embed_dim, num_channels, kernel) for kernel in kernel_sizes])
        
        mlp_input_dim = num_channels * len(kernel_sizes)
        self.mlp = SimpleMLP(input_dim=mlp_input_dim, output_dim=output_dim, hidden_dim=mlp_hidden_dims, dropout=dropout)

    @staticmethod
    def conv_and_pool(x, conv):
        conv_out = conv(x)
        pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
        return pooled
    
    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        conv_outputs = [self.conv_and_pool(x, conv) for conv in self.grouped_convs]

        x = torch.cat(conv_outputs, dim=1)
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


# class TransformerEncoder(nn.Module):
#     def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, mlp_hidden_dims, max_len, dropout=None, output_dim=1):
#         super(TransformerEncoder, self).__init__()
        
#         assert d_model % nhead == 0, "nheads must divide evenly into d_model"
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len)
#         encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, norm_first=True, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
#         self.decoder = SimpleMLP(input_dim=d_model, output_dim=output_dim, hidden_dim=mlp_hidden_dims, dropout=dropout)

#     def forward(self, src, src_mask=None, src_key_padding_mask=None):
#         src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
#         return self.decoder(output)


class BaseTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_len, dropout=None):
        super(BaseTransformerEncoder, self).__init__()
        
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, norm_first=True, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def encode(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)
        return self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)


class FlatTransformerEncoder(BaseTransformerEncoder):
    def __init__(self, mlp_hidden_dims, output_dim, *args, **kwargs):
        super(FlatTransformerEncoder, self).__init__(*args, **kwargs)
        self.decoder = SimpleMLP(input_dim=self.embedding.embedding_dim * kwargs.get("max_len"), output_dim=output_dim, hidden_dim=mlp_hidden_dims, dropout=kwargs.get("dropout"))

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output = self.encode(src, src_mask, src_key_padding_mask)
        output = output.flatten(start_dim=1)
        return self.decoder(output)


class MeanTransformerEncoder(BaseTransformerEncoder):
    def __init__(self, mlp_hidden_dims, output_dim, *args, **kwargs):
        super(MeanTransformerEncoder, self).__init__(*args, **kwargs)
        self.decoder = SimpleMLP(input_dim=self.embedding.embedding_dim, output_dim=output_dim, hidden_dim=mlp_hidden_dims, dropout=kwargs.get("dropout"))

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output = self.encode(src, src_mask, src_key_padding_mask)
        output = output.mean(dim=1)
        return self.decoder(output)


class CLSTransformerEncoder(BaseTransformerEncoder):
    def __init__(self, mlp_hidden_dims, output_dim, *args, **kwargs):
        kwargs["max_len"] += 1 # to account for CLS token
        super(CLSTransformerEncoder, self).__init__(*args, **kwargs)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding.embedding_dim))
        self.decoder = SimpleMLP(input_dim=self.embedding.embedding_dim, output_dim=output_dim, hidden_dim=mlp_hidden_dims, dropout=kwargs.get("dropout"))

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Embed the src token indices
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        
        # Repeat the cls_token for every item in the batch and concatenate it to src
        cls_tokens = self.cls_token.repeat(src.size(0), 1, 1)
        src = torch.cat([cls_tokens, src], dim=1)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Pass through transformer encoder
        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        # Extract the encoding corresponding to the cls_token
        output = output[:, 0, :]  # [B, E]
        
        return self.decoder(output)


class AttentionPoolingTransformerEncoder(BaseTransformerEncoder):
    def __init__(self, mlp_hidden_dims, output_dim, *args, **kwargs):
        super(AttentionPoolingTransformerEncoder, self).__init__(*args, **kwargs)
        self.attention_weight = nn.Linear(self.embedding.embedding_dim, 1)
        self.decoder = SimpleMLP(input_dim=self.embedding.embedding_dim, output_dim=output_dim, hidden_dim=mlp_hidden_dims, dropout=kwargs.get("dropout"))

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output = self.encode(src, src_mask, src_key_padding_mask)
        
        # Calculate attention scores. This will give a weight to each token in the sequence
        att_scores = self.attention_weight(output).squeeze(-1)
        att_probs = torch.softmax(att_scores, dim=1).unsqueeze(-1)
        
        # Weighted sum of the encoder outputs
        pooled_output = (output * att_probs).sum(dim=1)
        
        return self.decoder(pooled_output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Initialize pe with shape [1, max_len, d_model] for broadcasting
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Use broadcasting to add positional encoding
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super(Attention, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        # x should be (B, L, D)
        x, _ = self.attention(x, x, x)
        x = x.reshape(x.shape[0], -1)
        return x
    

class NeurLuxModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 max_len,
                 conv_channels=100,
                 lstm_out=32,
                 hidden_dim=32,
                 dropout=0.25,
                 output_dim=1):
        super(NeurLuxModel, self).__init__()
        self.__name__ = "NeurLux"
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=conv_channels,
                              kernel_size=4, padding=1)
        self.lstm = nn.LSTM(conv_channels, lstm_out, bidirectional=True,
                            batch_first=True)
        self.attention = Attention(embed_dim=lstm_out*2) # *2 since bidirectional
        attention_out = lstm_out*2*int((max_len-1)/4)
        self.fc1 = nn.Linear(attention_out, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = F.relu(x)
        # (B, L, MAX_LEN-1)
        x = F.max_pool1d(x, 4)
        # (B, L, (MAX_LEN-1)/4)
        # when MAX_LEN = 2048, (MAX_LEN-1)/4 = 511
        # when MAX_LEN = 87000, (MAX_LEN-1)/4 = 21749
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        # (B, L, D)
        # where D == H_out (32) * 2 (bidirectional)
        x = self.attention(x)
        # (B, L*D)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x