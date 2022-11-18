import json
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
        target_current_states, target_future_states,
        target_future_mask, target_indices, edge_indices,
        agents_timestamp,
    ) = map(list, zip(*samples))

    agents_polylines, num_agents = Polylines.collate(agents_polylines)
    roadgraph_polylines, num_roads = Polylines.collate(roadgraph_polylines)

    target_current_states = torch.stack(target_current_states, dim=0)
    target_future_states = torch.stack(target_future_states, dim=0)
    target_future_mask = torch.stack(target_future_mask, dim=0)

    target_indices = torch.as_tensor(target_indices, dtype=torch.int64)
    target_node_indices = target_indices.clone()

    num_nodes = [a + r for a, r in zip(num_agents, num_roads)]
    offset_agent = 0
    offset_node = 0
    for i in range(batch_size):
        target_indices[i] += offset_agent
        target_node_indices[i] += offset_node
        offset_agent += num_agents[i]
        offset_node += num_nodes[i]

    offset = 0
    edge_indices_list = []
    for edge_indices_i, num_nodes_i in zip(edge_indices, num_nodes):
        num_edges_i = edge_indices_i.shape[1]
        assert num_nodes_i ** 2 == num_edges_i
        edge_indices_list.append(edge_indices_i + offset)
        offset += num_nodes_i
    edge_indices = torch.cat(edge_indices_list, dim=-1)

    agents_timestamp = torch.cat(agents_timestamp, dim=0)

    return (
        agents_polylines, roadgraph_polylines,
        target_current_states, target_future_states, target_future_mask,
        target_indices, target_node_indices, edge_indices,
        agents_timestamp,
    )


class WaymoMotionDataset(Dataset):
    def __init__(
        self,
        root_path: Path,
        split: str,
        load_interval: int = 1,
    ):
        super().__init__()

        self.root_path = root_path
        self.split = split
        self.load_interval = load_interval

        with open(root_path / f"{split}.json") as f:
            samples = json.load(f)

        if split in ["training", "validation"]:
            samples = [
                (scenario_id, target_ids)
                for scenario_id, target_ids in samples.items()
            ]
            self.samples = samples[::load_interval]
        elif split in ["testing"]:
            assert load_interval == 1, load_interval
            self.samples = [
                (scenario_id, [target_id])
                for scenario_id, target_ids in samples.items()
                for target_id in target_ids
            ]
        else:
            raise ValueError(split)

    def __len__(self):
        return len(self.samples)

    def load_data(self, index: int):
        scenario_id, target_ids = self.samples[index]
        file_path = self.root_path / self.split / f"{scenario_id}.npz"

        data = np.load(file_path)

        agents_type = data["agents_type"]
        agents_states = data["agents_states"]
        agents_timestamp = data["agents_timestamp"]
        agents_states_mask = data["agents_states_mask"]
        agents_current_states = data["agents_current_states"]
        agents_future_states = data["agents_future_states"]
        agents_future_mask = data["agents_future_mask"]
        agents_agg_indices = data["agents_agg_indices"]
        roadgraph_type = data["roadgraph_type"]
        roadgraph_states = data["roadgraph_states"]
        compacted_roadgraph_id = data["compacted_roadgraph_id"]
        edge_indices = data["edge_indices"]

        # add agent type embedding
        agents_type = agents_type[agents_agg_indices]
        agents_type_one_hot = np.eye(5, dtype=np.float32)[agents_type]
        agents_states = np.concatenate(
            [agents_states, agents_type_one_hot], axis=-1
        )

        # add roadgraph type embedding
        roadgraph_type_one_hot = np.eye(20, dtype=np.float32)[roadgraph_type]
        roadgraph_states = np.concatenate(
            [roadgraph_states, roadgraph_type_one_hot], axis=-1
        )

        if self.split == "training":
            target_index = np.random.choice(target_ids, size=1)[0]
        else:
            target_index = target_ids[0]

        target_current_states = agents_current_states[target_index].copy()
        target_current_xy = target_current_states[:2].copy()
        target_current_yaw = target_current_states[2].copy()

        target_future_states = agents_future_states[target_index]
        target_future_mask = agents_future_mask[target_index]

        rot_matrix = np.asarray([
            [np.cos(target_current_yaw), -np.sin(target_current_yaw)],
            [np.sin(target_current_yaw), np.cos(target_current_yaw)],
        ])

        def transform(xy):
            return (xy - target_current_xy[None, :]) @ rot_matrix

        agents_states[:, :2] = transform(agents_states[:, :2])
        target_current_states[:2] = transform(target_current_states[:2])
        target_future_states[:, :2] = transform(target_future_states[:, :2])

        agents_states = torch.from_numpy(agents_states)
        agents_timestamp = torch.from_numpy(agents_timestamp)
        agents_states_mask = torch.from_numpy(agents_states_mask)
        agents_agg_indices = torch.from_numpy(agents_agg_indices)
        target_current_states = torch.from_numpy(target_current_states)
        target_future_states = torch.from_numpy(target_future_states)
        target_future_mask = torch.from_numpy(target_future_mask)

        agents_polylines = Polylines(
            features=agents_states,         # N, C
            agg_indices=agents_agg_indices, # N, (A, L, C) -> (A, C)
        )

        roadgraph_states[:, :2] = transform(roadgraph_states[:, :2])

        compacted_roadgraph_id = torch.from_numpy(compacted_roadgraph_id)
        roadgraph_states = torch.from_numpy(roadgraph_states)

        roadgraph_polylines = Polylines(
            features=roadgraph_states,          # N, C
            agg_indices=compacted_roadgraph_id, # N, (R, L, C) -> (R, C)
        )

        edge_indices = torch.from_numpy(edge_indices)

        return (
            agents_polylines, roadgraph_polylines,
            target_current_states, target_future_states,
            target_future_mask, target_index, edge_indices,
            agents_timestamp,
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

    def train_dataloader(self):
        dataset = WaymoMotionDataset(
            root_path=self.root_path,
            split="training",
            load_interval=self.train_interval,
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
            root_path=self.root_path,
            split="validation",
            load_interval=self.val_interval,
        )

        return DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        dataset = WaymoMotionDataset(
            root_path=self.root_path,
            split="testing",
            load_interval=self.test_interval,
        )

        return DataLoader(
            dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )


def test():
    dataset = WaymoMotionDataset(
        Path("data/waymo_open_dataset_motion_v_1_1_0/preprocessed"),
        split="training",
        load_interval=1,
    )
    dataset[0]
    print(len(dataset))

    dataset = WaymoMotionDataset(
        Path("data/waymo_open_dataset_motion_v_1_1_0/preprocessed"),
        split="validation",
        load_interval=1,
    )
    dataset[0]
    print(len(dataset))

    dataset = WaymoMotionDataset(
        Path("data/waymo_open_dataset_motion_v_1_1_0/preprocessed"),
        split="testing",
        load_interval=1,
    )
    dataset[0]
    print(len(dataset))


if __name__ == "__main__":
    test()
