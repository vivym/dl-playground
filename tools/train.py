from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.cli import LightningCLI, SaveConfigCallback


class WandbSaveConfigCallback(SaveConfigCallback):
    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Optional[str] = None
    ) -> None:
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    config_path = Path(logger.experiment.dir) / self.config_filename
                    self.parser.save(
                        self.config,
                        str(config_path),
                        skip_none=False,
                        overwrite=self.overwrite,
                        multifile=self.multifile,
                    )


def main():
    _ = LightningCLI(
        pl.LightningModule, pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        description="DDPM Playground",
        save_config_callback=WandbSaveConfigCallback,
        save_config_kwargs={"config_filename": "cli_config.yaml"},
        seed_everything_default=233,
    )


if __name__ == "__main__":
    main()
