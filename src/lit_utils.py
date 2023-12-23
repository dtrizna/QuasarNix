import os
import pickle
import numpy as np
from shutil import copyfile
from typing import Union

import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader
from torch import nn
from torch import cat, sigmoid

from .models import PyTorchLightningModel


class LitProgressBar(TQDMProgressBar):
    # to preserve progress bar after each epoch
    def on_train_epoch_end(self, *args, **kwargs):
        super().on_train_epoch_end(*args, **kwargs)
        print()


def configure_trainer(
        name: str,
        log_folder: str,
        epochs: int,
        device: str = "cpu",
        # how many times to check val set within a single epoch
        val_check_times: int = 2,
        log_every_n_steps: int = 10,
        monitor_metric: str = "val_tpr",
        early_stop_patience: Union[None, int] = 5,
        lit_sanity_steps: int = 1
):
    model_checkpoint = ModelCheckpoint(
        monitor=monitor_metric,
        save_top_k=1,
        mode="max",
        verbose=False,
        save_last=True,
        filename="{epoch}-tpr{val_tpr:.4f}-f1{val_f1:.4f}-acc{val_cc:.4f}"
    )
    callbacks = [ LitProgressBar(), model_checkpoint]

    if early_stop_patience is not None:
        early_stop = EarlyStopping(
            monitor=monitor_metric,
            patience=early_stop_patience,
            min_delta=0.0001,
            verbose=True,
            mode="max"
        )
        callbacks.append(early_stop)

    trainer = L.Trainer(
        num_sanity_val_steps=lit_sanity_steps,
        max_epochs=epochs,
        accelerator=device,
        devices=1,
        callbacks=callbacks,
        val_check_interval=1/val_check_times,
        log_every_n_steps=log_every_n_steps,
        logger=[
            CSVLogger(save_dir=log_folder, name=f"{name}_csv"),
            TensorBoardLogger(save_dir=log_folder, name=f"{name}_tb")
        ]
    )

    # Ensure folders for logging exist
    os.makedirs(os.path.join(log_folder, f"{name}_tb"), exist_ok=True)
    os.makedirs(os.path.join(log_folder, f"{name}_csv"), exist_ok=True)

    return trainer


def load_lit_model(
        model_file: str,
        pytorch_model: nn.Module,
        name: str,
        log_folder: str,
        epochs: int,
        device: str,
        lit_sanity_steps: int
):
    lightning_model = PyTorchLightningModel.load_from_checkpoint(checkpoint_path=model_file, model=pytorch_model)
    trainer = configure_trainer(name, log_folder, epochs, device=device, lit_sanity_steps=lit_sanity_steps)
    return trainer, lightning_model


def train_lit_model(
        X_train_loader: DataLoader,
        X_test_loader: DataLoader,
        pytorch_model: nn.Module,
        name: str,
        log_folder: str,
        epochs: int = 10,
        learning_rate: float = 1e-3,
        scheduler: Union[None, str] = None,
        scheduler_budget: Union[None, int] = None,
        model_file: Union[None, str] = None,
        device: str = "cpu",
        lit_sanity_steps: int = 1,
        early_stop_patience: int = 5
):
    lightning_model = PyTorchLightningModel(model=pytorch_model, learning_rate=learning_rate, scheduler=scheduler, scheduler_step_budget=scheduler_budget)
    trainer = configure_trainer(name, log_folder, epochs, device=device, lit_sanity_steps=lit_sanity_steps, early_stop_patience=early_stop_patience)

    print(f"[*] Training '{name}' model...")
    trainer.fit(lightning_model, X_train_loader, X_test_loader)

    if model_file is not None:
        # copy best checkpoint to the LOGS_DIR for further tests
        last_version_folder = [x for x in os.listdir(os.path.join(log_folder, name + "_csv")) if "version" in x][-1]
        checkpoint_path = os.path.join(log_folder, name + "_csv", last_version_folder, "checkpoints")
        best_checkpoint_name = [x for x in os.listdir(checkpoint_path) if x != "last.ckpt"][0]
        copyfile(os.path.join(checkpoint_path, best_checkpoint_name), model_file)

    return trainer, lightning_model


def predict_lit_model(
        loader: DataLoader, 
        trainer: L.Trainer, 
        lightning_model: PyTorchLightningModel, 
        decision_threshold: int = 0.5, 
        dump_logits: bool = False
) -> np.ndarray:
    """Get scores out of a loader."""
    y_pred_logits = trainer.predict(model=lightning_model, dataloaders=loader)
    y_pred = sigmoid(cat(y_pred_logits, dim=0)).numpy()
    y_pred = np.array([1 if x > decision_threshold else 0 for x in y_pred])
    if dump_logits:
        assert isinstance(dump_logits, str), "Please provide a path to dump logits: dump_logits='path/to/logits.pkl'"
        pickle.dump(y_pred_logits, open(dump_logits, "wb"))
    return y_pred
