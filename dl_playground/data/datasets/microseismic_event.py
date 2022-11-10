import json
from typing import List, Tuple, Callable, Optional
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


class MicroseismicEventDatasetImpl(Dataset):
    def __init__(
        self,
        root_path: Path,
        split: str,
        in_channels: List[str],
        repeat: int = 1,
        transform: Optional[Callable] = None
    ):
        super().__init__()

        self.root_path = root_path
        self.split = split
        self.in_channels = in_channels
        self.repeat = repeat
        self.transform = transform

        with open(root_path / f"{split}.json") as f:
            self.data_infos = json.load(f) * repeat

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index: int):
        file_name, label = self.data_infos[index]
        file_name = file_name.split(".")[0]

        imgs = []
        for in_channel in self.in_channels:
            img = Image.open(self.root_path / f"{file_name}_{in_channel}.jpg")
            img = np.asarray(img, dtype=np.float32) / 255.
            imgs.append(img)

        img = np.stack(imgs, axis=-1)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class MicroseismicEventDataset(pl.LightningDataModule):
    def __init__(
        self,
        root_path: str,
        in_channel_names: List[str],
        train_batch_size: int = 256,
        val_batch_size: int = 256,
        test_batch_size: int = 256,
        num_workers: int = 16,
        repeat: int = 1,
        mean: Tuple[float, ...] = (0.5, 0.5),
        std: Tuple[float, ...] = (0.25, 0.25),
    ):
        super().__init__()
        self.save_hyperparameters()

        self.root_path = Path(root_path)
        self.in_channels = in_channel_names
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.repeat = repeat
        self.mean = mean
        self.std = std

    def train_dataloader(self) -> DataLoader:
        transform = T.Compose([
            T.ToTensor(),
            T.Resize((224, 672)),
            T.Normalize(mean=self.mean, std=self.std),
        ])

        dataset = MicroseismicEventDatasetImpl(
            root_path=self.root_path,
            split="train",
            in_channels=self.in_channels,
            repeat=self.repeat,
            transform=transform,
        )

        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        transform = T.Compose([
            T.ToTensor(),
            T.Resize((224, 672)),
            T.Normalize(mean=self.mean, std=self.std),
        ])

        dataset = MicroseismicEventDatasetImpl(
            root_path=self.root_path,
            split="val",
            in_channels=self.in_channels,
            repeat=self.repeat,
            transform=transform,
        )

        return DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        transform = T.Compose([
            T.ToTensor(),
            T.Resize((224, 672)),
            T.Normalize(mean=self.mean, std=self.std),
        ])

        dataset = MicroseismicEventDatasetImpl(
            root_path=self.root_path,
            split="test",
            in_channels=self.in_channels,
            repeat=self.repeat,
            transform=transform,
        )

        return DataLoader(
            dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
