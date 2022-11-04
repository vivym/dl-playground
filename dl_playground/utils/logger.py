from pathlib import Path
from typing import Optional, Sequence, Union

from pytorch_lightning.loggers import WandbLogger as _WandbLogger


class WandbLogger(_WandbLogger):
    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: Optional[str] = None,
        version: Optional[str] = None,
        offline: Optional[bool] = False,
        dir: Optional[str] = None,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[str, bool] = False,
        experiment=None,
        prefix: str = "",
        entity: Optional[str] = None,
        job_type: Optional[str] = None,
        tags: Optional[Sequence] = None,
        group: Optional[str] = None,
        notes: Optional[str] = None,
        mode: Optional[str] = None,
        sync_tensorboard: Optional[bool] = False,
        monitor_gym: Optional[bool] = False,
        save_code: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=offline,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            entity=entity,
            job_type=job_type,
            tags=tags,
            group=group,
            notes=notes,
            mode=mode,
            sync_tensorboard=sync_tensorboard,
            monitor_gym=monitor_gym,
            save_code=save_code,
            **kwargs,
        )
