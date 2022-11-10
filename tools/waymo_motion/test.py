from pathlib import Path

from tqdm import tqdm
import torch

from dl_playground.data.datasets.waymo_motion import WaymoMotionDataset, collate_fn


def check_dataset(root_path: Path, is_training: bool = False):
    dataset = WaymoMotionDataset(
        root_path=root_path,
        is_training=is_training,
    )

    num_samples = len(dataset)

    for i in tqdm(range(8090, num_samples)):
        dataset[i]

def main():
    root_path = Path("data/waymo_open_dataset_motion_v_1_1_0/preprocessed")

    dataset = WaymoMotionDataset(
        root_path=root_path / "training",
        is_training=True,
    )

    num_samples = len(dataset)

    for i in [8098]:
        dataset[i]

    data = []
    samples = []
    for i in [10, 1000, 10000, 100000]:
        sample = dataset[i]
        (
            agents_polylines, roadgraph_polylines,
            target_future_states, target_future_mask,
            target_index, edge_indices,
        ) = sample
        samples.append(sample)

        data.append((
            agents_polylines.features, agents_polylines.agg_indices,
            roadgraph_polylines.features, roadgraph_polylines.agg_indices,
            target_future_states, target_future_mask,
            target_index, edge_indices,
        ))

    (
        agents_polylines, roadgraph_polylines,
        target_future_states, target_future_mask,
        target_indices, target_node_indices, edge_indices,
    ) = collate_fn(samples)

    batched_data = (
        agents_polylines.features, agents_polylines.agg_indices, agents_polylines.batch_indices,
        roadgraph_polylines.features, roadgraph_polylines.agg_indices, roadgraph_polylines.batch_indices,
        target_future_states, target_future_mask,
        target_indices, target_node_indices, edge_indices,
    )

    torch.save((data, batched_data), "wandb/data.pth")


if __name__ == "__main__":
    main()
