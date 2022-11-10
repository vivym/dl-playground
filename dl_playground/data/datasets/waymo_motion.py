from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class Polylines:
    features: torch.Tensor
    agg_indices: Optional[torch.Tensor] = None
    batch_indices: Optional[torch.Tensor] = None
    batch_size: Optional[int] = None

    @classmethod
    def collate(
        cls,
        polylines_list: List["Polylines"],
        offsets: Optional[List[int]] = None,
    ):
        features = torch.cat([
            polylines.features
            for polylines in polylines_list
        ], dim=0)

        agg_indices = []
        batch_indices = []
        offset = 0
        num_polylines_list = []
        for i, polylines in enumerate(polylines_list):
            if offsets is not None:
                offset = offsets[i]
            agg_indices.append(polylines.agg_indices + offset)
            num_polylines = polylines.agg_indices.max().item() + 1
            num_polylines_list.append(num_polylines)
            offset += num_polylines
            batch_indices.append(torch.full((num_polylines,), i, dtype=torch.int64))

        agg_indices = torch.cat(agg_indices, dim=0)
        batch_indices = torch.cat(batch_indices, dim=0)

        return Polylines(
            features=features,
            agg_indices=agg_indices,
            batch_indices=batch_indices,
            batch_size=len(polylines_list)
        ), num_polylines_list


def collate_fn(samples):
    (
        agents_polylines, agents_polylines_mask,
        roadgraph_polylines, roadgraph_polylines_mask,
        targets, targets_mask,
        tracks_to_predict, objects_of_interest,
        agents_motion,
    ) = map(list, zip(*samples))

    batch_size = len(roadgraph_polylines)
    max_num_roads = np.max([polyline.shape[0] for polyline in roadgraph_polylines])
    max_road_length = np.max([polyline.shape[1] for polyline in roadgraph_polylines])
    roadgraph_vec_channels = roadgraph_polylines[0].shape[2]

    roadgraph_polylines_padded = torch.zeros(
        batch_size, max_num_roads, max_road_length, roadgraph_vec_channels
    )
    roadgraph_polylines_mask_padded = torch.zeros(
        batch_size, max_num_roads, max_road_length, dtype=torch.bool
    )
    for batch_ind in range(batch_size):
        num_roads, road_length, _ = roadgraph_polylines[batch_ind].shape
        roadgraph_polylines_padded[batch_ind, :num_roads, :road_length, :] = roadgraph_polylines[batch_ind]
        roadgraph_polylines_mask_padded[batch_ind, :num_roads, :road_length] = roadgraph_polylines_mask[batch_ind]

    return (
        torch.stack(agents_polylines), torch.stack(agents_polylines_mask),
        roadgraph_polylines_padded, roadgraph_polylines_mask_padded,
        torch.stack(targets), torch.stack(targets_mask),
        torch.stack(tracks_to_predict), torch.stack(objects_of_interest),
        torch.stack(agents_motion),
    )


class WaymoMotionDataset(Dataset):
    def __init__(
        self,
        root_path: Path,
        load_interval: int = 1,
        is_training: bool = False
    ):
        super().__init__()

        self.root_path = root_path
        self.load_interval = load_interval
        self.is_training = is_training

        self.paths = list(root_path.glob("*.npz"))
        self.paths = self.paths[::load_interval]

    def __len__(self):
        return len(self.paths)

    def load_data(self, index: int):
        file_path = self.paths[index]

        data = np.load(file_path)

        _, num_past_steps = data["state/past/x"].shape
        _, num_current_steps = data["state/current/x"].shape

        agents_type_one_hot = np.eye(5)[data["state/type"].astype(np.int64).flatten()].astype(np.float32)

        past_agents = np.concatenate([
            data["state/past/x"][:, :, None],
            data["state/past/y"][:, :, None],
            data["state/past/z"][:, :, None],
            np.tile(agents_type_one_hot[:, None, :], (1, num_past_steps, 1)),
        ], axis=2)

        current_agents = np.concatenate([
            data["state/current/x"][:, :, None],
            data["state/current/y"][:, :, None],
            data["state/current/z"][:, :, None],
            np.tile(agents_type_one_hot[:, None, :], (1, num_current_steps, 1)),
        ], axis=2)

        agents_polylines = np.concatenate([current_agents, past_agents], axis=1)
        agents_polylines_mask = np.concatenate([
            data["state/current/valid"], data["state/past/valid"]
        ], axis=1).astype(bool)

        roadgraph_type = data["roadgraph_samples/type"].astype(np.int64).flatten()
        roadgraph_type_one_hot = np.eye(20)[roadgraph_type].astype(np.float32)

        roadgraph_id_set = set([
            ind for ind in data["roadgraph_samples/id"].reshape(-1).tolist() if ind > 0
        ])
        roadgraph_vecs = np.concatenate([
            data["roadgraph_samples/xyz"],
            data["roadgraph_samples/dir"],
            roadgraph_type_one_hot,
        ], axis=1)
        roadgraph_vecs_mask = data["roadgraph_samples/valid"].astype(bool).reshape(-1)

        _, roadgraph_vecs_channels = roadgraph_vecs.shape

        num_roads = len(roadgraph_id_set)
        max_road_length = np.max([
            np.sum(data["roadgraph_samples/id"].flatten() == ind) for ind in roadgraph_id_set
        ])

        roadgraph_polylines = np.zeros(shape=(num_roads, max_road_length, roadgraph_vecs_channels))
        roadgraph_polylines_mask = np.zeros(shape=(num_roads, max_road_length), dtype=bool)

        for i, ind in enumerate(roadgraph_id_set):
            indices = data["roadgraph_samples/id"].flatten() == ind
            polyline_length = np.sum(indices)
            roadgraph_polylines[i, :polyline_length, :] = roadgraph_vecs[indices, :]
            roadgraph_polylines_mask[i, :polyline_length] = roadgraph_vecs_mask[indices]

        _, num_future_steps = data["state/future/x"].shape
        targets = np.dstack([
            data["state/future/x"],
            data["state/future/y"],
            data["state/future/z"],
            np.tile(data["state/type"].reshape(-1, 1), (1, num_future_steps)),
        ])
        targets_mask = data["state/future/valid"].astype(bool)

        tracks_to_predict = data["state/tracks_to_predict"]
        objects_of_interest = data["state/objects_of_interest"]

        agents_motion = np.concatenate([
            data["state/current/velocity_x"],
            data["state/current/velocity_y"],
            data["state/current/vel_yaw"],
        ], axis=-1)

        rng = np.random.default_rng()
        if np.sum(objects_of_interest == 1) < 2:
            if np.sum(tracks_to_predict == 1) < 2:
                return None
            else:
                indices = rng.choice(np.argwhere(tracks_to_predict == 1), size=2, replace=False, p=None)
                objects_of_interest[indices] = 1

        origin = np.mean(
            agents_polylines[np.argwhere(objects_of_interest == 1).reshape(-1), 0, :3], axis=0
        )

        agents_polylines[agents_polylines_mask, :3] = agents_polylines[agents_polylines_mask, :3] - origin
        roadgraph_polylines[roadgraph_polylines_mask, :3] = roadgraph_polylines[roadgraph_polylines_mask, :3] - origin
        targets[targets_mask, :3] = targets[targets_mask, :3] - origin

        return (
            torch.as_tensor(agents_polylines),
            torch.as_tensor(agents_polylines_mask),
            torch.as_tensor(roadgraph_polylines),
            torch.as_tensor(roadgraph_polylines_mask),
            torch.as_tensor(targets),
            torch.as_tensor(targets_mask),
            torch.as_tensor(tracks_to_predict),
            torch.as_tensor(objects_of_interest),
            torch.as_tensor(agents_motion),
        )

    def __getitem__(self, index: int):
        while True:
            data = self.load_data(index)
            if data is not None:
                return data

            index += 1
            if index >= len(self.paths):
                index = 0


class WaymoMotionDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        root_path: str,
        train_interval: int = 1,
        val_interval: int = 1,
        test_interval: int = 1,
        train_batch_size: int = 64,
        val_batch_size: int = 64,
        test_batch_size: int = 64,
        num_workers: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.root_path = Path(root_path)
        self.train_interval = train_interval
        self.val_interval = val_interval
        self.test_interval = test_interval
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        self.train_path = self.root_path / "preprocessed" / "training"
        self.val_path = self.root_path / "preprocessed" / "validation"

    def train_dataloader(self):
        dataset = WaymoMotionDataset(
            root_path=self.train_path,
            load_interval=self.train_interval,
            is_training=True,
        )

        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        dataset = WaymoMotionDataset(
            root_path=self.val_path,
            load_interval=self.val_interval,
            is_training=False,
        )

        return DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )


def test():
    dataset = WaymoMotionDataset(
        Path("data/waymo_open_dataset_motion_v_1_1_0/preprocessed/training"),
        load_interval=10,
        is_training=True,
    )
    data = dataset[0]
    # print(data)


if __name__ == "__main__":
    test()
