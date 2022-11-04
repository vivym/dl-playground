import pytorch_lightning as pl


class ImageFolder(pl.LightningDataModule):
    def __init__(
        self
    ):
        super().__init__()
        self.save_hyperparameters()
