import argparse
from pathlib import Path

import numpy as np
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


def preprocess(
    root_path: Path,
    out_path: Path,
    num_shard: int = 1,
    shard_index: int = 0,
) -> None:
    if not out_path.exists():
        out_path.mkdir(parents=True)

    feature_descriptions = create_feature_descriptions()

    dataset = tf.data.TFRecordDataset(
        list(root_path.glob("*tfexample.tfrecord*")),
        num_parallel_reads=1,
    )
    # dataset.shard(num_shard, shard_index)

    for i, raw_data in tqdm(enumerate(dataset.as_numpy_iterator())):
        data = tf.io.parse_single_example(raw_data, feature_descriptions)
        new_data = {}
        for key, value in data.items():
            if key == "scenario/id":
                new_data[key] = value.numpy()[0].decode("utf8")
            else:
                new_data[key] = value.numpy()

        scenario_id = new_data["scenario/id"]
        np.savez_compressed(out_path / f"{scenario_id}.npz", **new_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-path", type=str, default="data/waymo_open_dataset_motion_v_1_1_0"
    )
    parser.add_argument(
        "--out-path", type=str, default="data/waymo_open_dataset_motion_v_1_1_0/preprocessed"
    )
    parser.add_argument("--num-workers", type=int, default=64)
    args = parser.parse_args()

    root_path, out_path = Path(args.root_path), Path(args.out_path)

    preprocess(
        root_path=root_path / "uncompressed" / "tf_example" / "training",
        out_path=out_path / "training",
    )

    preprocess(
        root_path=root_path / "uncompressed" / "tf_example" / "validation",
        out_path=out_path / "validation",
    )


if __name__ == "__main__":
    main()
