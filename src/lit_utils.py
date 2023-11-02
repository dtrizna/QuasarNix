import os
from shutil import copyfile
from typing import Union

import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

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


def load_lit_model(model_file, pytorch_model, name, log_folder, epochs, device, lit_sanity_steps):
    lightning_model = PyTorchLightningModel.load_from_checkpoint(checkpoint_path=model_file, model=pytorch_model)
    trainer = configure_trainer(name, log_folder, epochs, device=device, lit_sanity_steps=lit_sanity_steps)
    return trainer, lightning_model


def train_lit_model(
        X_train_loader,
        X_test_loader,
        pytorch_model,
        name,
        log_folder,
        epochs=10,
        learning_rate=1e-3,
        scheduler=None,
        scheduler_budget=None,
        model_file=None,
        device="cpu",
        lit_sanity_steps=1
):
    lightning_model = PyTorchLightningModel(model=pytorch_model, learning_rate=learning_rate, scheduler=scheduler, scheduler_step_budget=scheduler_budget)
    trainer = configure_trainer(name, log_folder, epochs, device=device, lit_sanity_steps=lit_sanity_steps)

    print(f"[*] Training '{name}' model...")
    trainer.fit(lightning_model, X_train_loader, X_test_loader)

    if model_file is not None:
        # copy best checkpoint to the LOGS_DIR for further tests
        last_version_folder = [x for x in os.listdir(os.path.join(log_folder, name + "_csv")) if "version" in x][-1]
        checkpoint_path = os.path.join(log_folder, name + "_csv", last_version_folder, "checkpoints")
        best_checkpoint_name = [x for x in os.listdir(checkpoint_path) if x != "last.ckpt"][0]
        copyfile(os.path.join(checkpoint_path, best_checkpoint_name), model_file)

    return trainer, lightning_model
