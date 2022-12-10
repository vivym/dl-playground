import argparse
import json
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import torch
import tensorflow as tf
from tqdm import tqdm


def create_feature_descriptions():
    roadgraph_features = {
        "roadgraph_samples/dir":
            tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
        "roadgraph_samples/id":
            tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
        "roadgraph_samples/type":
            tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
        "roadgraph_samples/valid":
            tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
        "roadgraph_samples/xyz":
            tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
    }

    state_features = {
        "state/id":
            tf.io.FixedLenFeature([128], tf.float32, default_value=None),
        "state/type":
            tf.io.FixedLenFeature([128], tf.float32, default_value=None),
        "state/is_sdc":
            tf.io.FixedLenFeature([128], tf.int64, default_value=None),
        "state/tracks_to_predict":
            tf.io.FixedLenFeature([128], tf.int64, default_value=None),
        "scenario/id":
            tf.io.FixedLenFeature([1], tf.string, default_value=None),
        "state/current/bbox_yaw":
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        "state/current/height":
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        "state/current/length":
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        "state/current/timestamp_micros":
            tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
        "state/current/valid":
            tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
        "state/current/vel_yaw":
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        "state/current/velocity_x":
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        "state/current/velocity_y":
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        "state/current/width":
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        "state/current/x":
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        "state/current/y":
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        "state/current/z":
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        "state/future/bbox_yaw":
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        "state/future/height":
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        "state/future/length":
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        "state/future/timestamp_micros":
            tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
        "state/future/valid":
            tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
        "state/future/vel_yaw":
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        "state/future/velocity_x":
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        "state/future/velocity_y":
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        "state/future/width":
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        "state/future/x":
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        "state/future/y":
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        "state/future/z":
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        "state/past/bbox_yaw":
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        "state/past/height":
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        "state/past/length":
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        "state/past/timestamp_micros":
            tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
        "state/past/valid":
            tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
        "state/past/vel_yaw":
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        "state/past/velocity_x":
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        "state/past/velocity_y":
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        "state/past/width":
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        "state/past/x":
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        "state/past/y":
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        "state/past/z":
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        "state/objects_of_interest":
            tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    }

    traffic_light_features = {
        "traffic_light_state/current/state":
            tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
        "traffic_light_state/current/valid":
            tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
        "traffic_light_state/current/x":
            tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
        "traffic_light_state/current/y":
            tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
        "traffic_light_state/current/z":
            tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
        "traffic_light_state/past/state":
            tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
        "traffic_light_state/past/valid":
            tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
        "traffic_light_state/past/x":
            tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
        "traffic_light_state/past/y":
            tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
        "traffic_light_state/past/z":
            tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
        "traffic_light_state/current/id":
            tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
        "traffic_light_state/past/id":
            tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
        "traffic_light_state/past/state":
            tf.io.FixedLenFeature([10,16],tf.int64,default_value=None),
        "traffic_light_state/current/state":
            tf.io.FixedLenFeature([1,16],tf.int64,default_value=None),
        "traffic_light_state/past/timestamp_micros":
            tf.io.FixedLenFeature([10],tf.int64,default_value=None),
        "traffic_light_state/current/timestamp_micros":
            tf.io.FixedLenFeature([1],tf.int64,default_value=None),
    }

    feature_descriptions = {}
    feature_descriptions.update(roadgraph_features)
    feature_descriptions.update(state_features)
    feature_descriptions.update(traffic_light_features)
    return feature_descriptions


def to_numpy(
    root_path: Path,
    out_path: Path,
    num_shard: int = 1,
    shard_index: int = 0,
) -> None:
    if not out_path.exists():
        out_path.mkdir(parents=True)

    feature_descriptions = create_feature_descriptions()

    for record_path in tqdm(sorted(root_path.glob("*tfexample.tfrecord*"))):
        print("record_path", record_path)
        if record_path.name not in ["training_tfexample.tfrecord-00223-of-01000"]:
            continue

        dataset = tf.data.TFRecordDataset(
            [record_path],
            num_parallel_reads=1,
        )
        # dataset.shard(num_shard, shard_index)

        for raw_data in tqdm(dataset.as_numpy_iterator()):
            data = tf.io.parse_single_example(raw_data, feature_descriptions)
            new_data = {}
            for key, value in data.items():
                if key == "scenario/id":
                    new_data[key] = value.numpy()[0].decode("utf8")
                else:
                    new_data[key] = value.numpy()

            scenario_id = new_data["scenario/id"]
            np.savez_compressed(out_path / f"{scenario_id}.npz", **new_data)
        print("done")


def preprocess_worker(file_path: Path, out_dir: Path, is_training: bool):
    data = np.load(file_path)

    agents_past_mask = data["state/past/valid"]
    agents_current_mask = data["state/current/valid"]

    agents_states_mask = np.concatenate([
        agents_past_mask, agents_current_mask
    ], axis=-1).astype(bool)
    agents_future_mask = data["state/future/valid"].astype(bool)

    valid_agents_mask = agents_states_mask.any(-1)
    num_valid_agents = valid_agents_mask.sum()

    tracks_to_predict = data["state/tracks_to_predict"].astype(bool)

    agents_past_mask = agents_past_mask[valid_agents_mask]
    agents_current_mask = agents_current_mask[valid_agents_mask]
    agents_states_mask = agents_states_mask[valid_agents_mask]
    agents_future_mask = agents_future_mask[valid_agents_mask]
    tracks_to_predict = tracks_to_predict[valid_agents_mask]

    assert np.sum(tracks_to_predict) > 0

    agents_id = data["state/id"].astype(np.int64)[valid_agents_mask]
    agents_type = data["state/type"].astype(np.int64)[valid_agents_mask]

    agents_past_x = data["state/past/x"][valid_agents_mask]
    agents_past_y = data["state/past/y"][valid_agents_mask]
    agents_past_yaw = data["state/past/bbox_yaw"][valid_agents_mask]
    agents_past_vel_x = data["state/past/velocity_x"][valid_agents_mask]
    agents_past_vel_y = data["state/past/velocity_y"][valid_agents_mask]
    agents_past_vel_yaw = data["state/past/vel_yaw"][valid_agents_mask]

    agents_current_x = data["state/current/x"][valid_agents_mask]
    agents_current_y = data["state/current/y"][valid_agents_mask]
    agents_current_yaw = data["state/current/bbox_yaw"][valid_agents_mask]
    agents_current_vel_x = data["state/current/velocity_x"][valid_agents_mask]
    agents_current_vel_y = data["state/current/velocity_y"][valid_agents_mask]
    agents_current_vel_yaw = data["state/current/vel_yaw"][valid_agents_mask]

    agents_future_x = data["state/future/x"][valid_agents_mask]
    agents_future_y = data["state/future/y"][valid_agents_mask]
    agents_future_yaw = data["state/future/bbox_yaw"][valid_agents_mask]
    agents_future_vel_x = data["state/future/velocity_x"][valid_agents_mask]
    agents_future_vel_y = data["state/future/velocity_y"][valid_agents_mask]
    agents_future_vel_yaw = data["state/future/vel_yaw"][valid_agents_mask]

    agents_past_states = np.concatenate([
        agents_past_x[:, :, None],
        agents_past_y[:, :, None],
        agents_past_yaw[:, :, None],
        agents_past_vel_x[:, :, None],
        agents_past_vel_y[:, :, None],
        agents_past_vel_yaw[:, :, None],
    ], axis=-1)

    agents_current_states = np.concatenate([
        agents_current_x[:, :, None],
        agents_current_y[:, :, None],
        agents_current_yaw[:, :, None],
        agents_current_vel_x[:, :, None],
        agents_current_vel_y[:, :, None],
        agents_current_vel_yaw[:, :, None],
    ], axis=-1)

    agents_future_states = np.concatenate([
        agents_future_x[:, :, None],
        agents_future_y[:, :, None],
        agents_future_yaw[:, :, None],
        agents_future_vel_x[:, :, None],
        agents_future_vel_y[:, :, None],
        agents_future_vel_yaw[:, :, None],
    ], axis=-1)

    # A, T, C
    agents_states = np.concatenate([agents_past_states, agents_current_states], axis=1)
    agents_timestamp = np.tile(np.arange(-10, 1)[None, :], (num_valid_agents, 1))
    # N, C
    agents_states = agents_states[agents_states_mask]
    agents_timestamp = agents_timestamp[agents_states_mask]

    # A, C
    agents_current_states = agents_current_states[:, 0, :].copy()

    agents_agg_indices = agents_states_mask.nonzero()[0]

    roadgraph_states_mask = data["roadgraph_samples/valid"].astype(bool).flatten()

    roadgraph_id = data["roadgraph_samples/id"].astype(np.int64).flatten()[roadgraph_states_mask]
    roadgraph_type = data["roadgraph_samples/type"].astype(np.int64).flatten()[roadgraph_states_mask]

    roadgraph_xyz = data["roadgraph_samples/xyz"][roadgraph_states_mask]
    roadgraph_dir = data["roadgraph_samples/dir"][roadgraph_states_mask]

    roadgraph_states = np.concatenate([
        roadgraph_xyz,
        roadgraph_dir,
    ], axis=-1)

    unique_roadgraph_id, compacted_roadgraph_id = np.unique(roadgraph_id, return_inverse=True)
    num_valid_roads = unique_roadgraph_id.shape[0]

    num_nodes = num_valid_agents + num_valid_roads
    range_indices = torch.arange(num_nodes, dtype=torch.int64)
    u, v = torch.meshgrid(range_indices, range_indices, indexing="ij")
    edge_indices = torch.stack([u, v], dim=0)
    edge_indices = edge_indices.flatten(1)
    edge_indices = edge_indices.numpy()

    # if is_training:
    #     predictable_mask = agents_future_mask.any(-1) & agents_current_mask.any(-1)
    # else:
    #     # TODO: regenerate data
    #     predictable_mask = tracks_to_predict & agents_current_mask.any(-1)
    predictable_mask = tracks_to_predict & agents_current_mask.any(-1)
    target_ids = predictable_mask.nonzero()[0]

    np.savez_compressed(
        out_dir / file_path.name,
        agents_id=agents_id,
        agents_type=agents_type,
        agents_states=agents_states,
        agents_timestamp=agents_timestamp,
        agents_states_mask=agents_states_mask,
        agents_current_states=agents_current_states,
        agents_future_states=agents_future_states,
        agents_future_mask=agents_future_mask,
        agents_agg_indices=agents_agg_indices,
        roadgraph_id=roadgraph_id,
        roadgraph_type=roadgraph_type,
        roadgraph_states=roadgraph_states,
        compacted_roadgraph_id=compacted_roadgraph_id,
        num_valid_agents=num_valid_agents,
        num_valid_roads=num_valid_roads,
        edge_indices=edge_indices,
        target_ids=target_ids,
    )

    return file_path.stem, target_ids.tolist()


def preprocess_worker_wrapper(args):
    return preprocess_worker(*args)


def preprocess(root_path: Path, out_path: Path, num_workers: int, is_training: bool):
    samples = []
    with Pool(processes=num_workers) as p:
        args = [
            (file_path, out_path, is_training)
            for file_path in root_path.glob("*.npz")
        ]
        # for arg in args:
        #     preprocess_worker_wrapper(arg)

        samples = list(tqdm(
            p.imap_unordered(preprocess_worker_wrapper, args, chunksize=num_workers * 2),
            total=len(args)
        ))

    return {
        scenario_id: target_ids
        for scenario_id, target_ids in samples
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-path", type=str, default="data/waymo_open_dataset_motion_v_1_1_0/"
    )
    parser.add_argument(
        "--out-path", type=str, default="data/waymo_open_dataset_motion_v_1_1_0/preprocessed_npz"
    )
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()

    root_path, out_path = Path(args.root_path), Path(args.out_path)

    if not out_path.exists():
        out_path.mkdir(parents=True)

    to_numpy(
        root_path=root_path / "uncompressed" / "tf_example" / "training",
        out_path=out_path / "training",
    )

    # to_numpy(
    #     root_path=root_path / "uncompressed" / "tf_example" / "validation",
    #     out_path=out_path / "validation",
    # )


if __name__ == "__main__":
    main()
