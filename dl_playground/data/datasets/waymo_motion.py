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
    batch_size = len(samples)
    (
        agents_polylines, roadgraph_polylines,
        target_future_states, target_future_mask,
        target_indices, edge_indices,
    ) = map(list, zip(*samples))

    agents_polylines, num_agents = Polylines.collate(agents_polylines)
    roadgraph_polylines, num_roads = Polylines.collate(roadgraph_polylines)

    target_future_states = torch.stack(target_future_states, dim=0)
    target_future_mask = torch.stack(target_future_mask, dim=0)

    target_indices = torch.as_tensor(target_indices, dtype=torch.int64)

    target_node_indices = target_indices.clone()
    num_nodes = [a + r for a, r in zip(num_agents, num_roads)]
    offset = 0
    for i in range(batch_size):
        target_node_indices[i] += offset
        offset += num_nodes[i]

    offset = 0
    edge_indices_list = []
    for edge_indices_i, num_nodes_i in zip(edge_indices, num_nodes):
        num_edges_i = edge_indices_i.shape[1]
        assert num_nodes_i ** 2 == num_edges_i
        edge_indices_list.append(edge_indices_i + offset)
        offset += num_nodes_i
    edge_indices = torch.cat(edge_indices_list, dim=-1)

    return (
        agents_polylines, roadgraph_polylines,
        target_future_states, target_future_mask,
        target_indices, target_node_indices, edge_indices,
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

        agents_states_mask = np.concatenate([
            data["state/past/valid"], data["state/current/valid"]
        ], axis=-1).astype(bool)
        agents_future_mask = data["state/future/valid"].astype(bool)

        # print("valid", data["state/current/valid"].sum())

        # assert (data["state/current/x"][data["state/current/valid"].astype(bool)] > 0).all(), (
        #     data["state/current/x"][data["state/current/valid"].astype(bool)]
        # )
        # assert (data["state/current/x"][data["state/current/valid"].astype(bool).reshape(-1)] > 0).all(), (
        #     data["state/current/x"][data["state/current/valid"].astype(bool).reshape(-1)]
        # )

        valid_agents_mask = agents_states_mask[:, -1]
        invalid_agents_mask = ~valid_agents_mask
        num_valid_agents = valid_agents_mask.sum()

        tracks_to_predict = data["state/tracks_to_predict"].astype(bool)
        tracks_to_predict[invalid_agents_mask] = False

        agents_states_mask = agents_states_mask[valid_agents_mask]
        agents_future_mask = agents_future_mask[valid_agents_mask]
        tracks_to_predict = tracks_to_predict[valid_agents_mask]

        assert agents_future_mask[tracks_to_predict].any(-1).all()

        if self.is_training:
            valid_target_agents_mask = agents_future_mask.any(-1)
            if np.sum(valid_target_agents_mask) == 0:
                return None

            target_index = np.random.choice(
                np.argwhere(valid_target_agents_mask).reshape(-1), size=1, replace=False
            )[0]
        else:
            if np.sum(tracks_to_predict) == 0:
                return None
            target_index = np.random.choice(
                np.argwhere(tracks_to_predict).reshape(-1), size=1, replace=False
            )[0]

        num_past_steps = data["state/past/x"].shape[1]
        num_current_steps = data["state/current/x"].shape[1]
        num_future_steps = data["state/future/x"].shape[1]

        agents_type_one_hot = np.eye(5)[data["state/type"].astype(np.int64).flatten()].astype(np.float32)
        agents_past_states = np.concatenate([
            data["state/past/x"][:, :, None],
            data["state/past/y"][:, :, None],
            data["state/past/bbox_yaw"][:, :, None],
            data["state/past/velocity_x"][:, :, None],
            data["state/past/velocity_y"][:, :, None],
            data["state/past/vel_yaw"][:, :, None],
            np.tile(agents_type_one_hot[:, None, :], (1, num_past_steps, 1)),
        ], axis=-1)

        agents_current_states = np.concatenate([
            data["state/current/x"][:, :, None],
            data["state/current/y"][:, :, None],
            data["state/current/bbox_yaw"][:, :, None],
            data["state/current/velocity_x"][:, :, None],
            data["state/current/velocity_y"][:, :, None],
            data["state/current/vel_yaw"][:, :, None],
            np.tile(agents_type_one_hot[:, None, :], (1, num_current_steps, 1)),
        ], axis=-1)

        agents_states = np.concatenate([agents_past_states, agents_current_states], axis=1)

        # T, C
        target_future_states = np.concatenate([
            data["state/future/x"][valid_agents_mask, :, None],
            data["state/future/y"][valid_agents_mask, :, None],
            data["state/future/bbox_yaw"][valid_agents_mask, :, None],
            data["state/future/velocity_x"][valid_agents_mask, :, None],
            data["state/future/velocity_y"][valid_agents_mask, :, None],
            data["state/future/vel_yaw"][valid_agents_mask, :, None],
            np.tile(agents_type_one_hot[valid_agents_mask, None, :], (num_future_steps, 1)),
        ], axis=-1)
        target_future_states = target_future_states[target_index]

        # A, T, C
        agents_states = agents_states[valid_agents_mask]

        target_current_xy = agents_states[target_index, -1, :2]
        target_current_yaw = agents_states[target_index, -1, 2]

        rot_matrix = np.asarray([
            [np.cos(target_current_yaw), -np.sin(target_current_yaw)],
            [np.sin(target_current_yaw), np.cos(target_current_yaw)],
        ])

        # N, C
        agents_states = agents_states[agents_states_mask]
        agents_states[:, :2] = (agents_states[:, :2] - target_current_xy[None, :]) @ rot_matrix

        target_future_states[:, :2] = (target_future_states[:, :2] - target_current_xy[None, :]) @ rot_matrix

        agents_states = torch.from_numpy(agents_states)
        agents_states_mask = torch.from_numpy(agents_states_mask)
        target_future_states = torch.from_numpy(target_future_states)
        target_future_mask = torch.from_numpy(agents_future_mask[target_index])

        if target_future_states[target_future_mask, :2].max() > 1e+3:
            print("target_index", target_index)
            print("agents_states_mask", agents_states_mask[target_index])

            print("target_current_xy", target_current_xy)
            print("target_future_states", target_future_states[:5, :2])

            assert target_future_states[target_future_mask, :2].max() < 1e+3, (
                target_future_states[target_future_mask, :2], target_future_states[target_future_mask, :2].shape, target_future_mask
            )

        agents_polylines = Polylines(
            features=agents_states,                                     # N, C
            agg_indices=agents_states_mask.nonzero(as_tuple=True)[0],   # N, (A, L, C) -> (A, C)
        )

        roadgraph_type = data["roadgraph_samples/type"].astype(np.int64).flatten()
        roadgraph_type_one_hot = np.eye(20)[roadgraph_type].astype(np.float32)

        roadgraph_states = np.concatenate([
            data["roadgraph_samples/xyz"],
            data["roadgraph_samples/dir"],
            roadgraph_type_one_hot,
        ], axis=-1)
        roadgraph_states_mask = data["roadgraph_samples/valid"].astype(bool).reshape(-1)

        roadgraph_states = roadgraph_states[roadgraph_states_mask]
        roadgraph_states[:, :2] = (roadgraph_states[:, :2] - target_current_xy[None, :]) @ rot_matrix

        roadgraph_id = data["roadgraph_samples/id"].reshape(-1)
        roadgraph_id = roadgraph_id[roadgraph_states_mask]
        unique_roadgraph_id, roadgraph_id = np.unique(roadgraph_id, return_inverse=True)
        num_valid_roads = unique_roadgraph_id.shape[0]

        roadgraph_states = torch.from_numpy(roadgraph_states)
        roadgraph_id = torch.from_numpy(roadgraph_id)

        roadgraph_polylines = Polylines(
            features=roadgraph_states,  # N, C
            agg_indices=roadgraph_id,   # N, (R, L, C) -> (R, C)
        )

        # num_valid_agents + num_valid_roads
        num_nodes = num_valid_agents + num_valid_roads
        range_indices = torch.arange(num_nodes, dtype=torch.int64)
        u, v = torch.meshgrid(range_indices, range_indices, indexing="ij")
        edge_indices = torch.stack([u, v], dim=0)
        edge_indices = edge_indices.flatten(1)

        return (
            agents_polylines, roadgraph_polylines,
            target_future_states, target_future_mask,
            target_index, edge_indices,
        )

    def __getitem__(self, index: int):
        while True:
            data = self.load_data(index)
            if data is not None:
                return data

            # print("skip", index)
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
