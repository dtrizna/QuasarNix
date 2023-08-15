from lightning.pytorch.callbacks import TQDMProgressBar

class LitProgressBar(TQDMProgressBar):
    # to preserve progress bar after each epoch
    def on_train_epoch_end(self, *args, **kwargs):
        super().on_train_epoch_end(*args, **kwargs)
        print()
