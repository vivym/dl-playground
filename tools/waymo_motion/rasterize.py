import argparse
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm


def rasterize(
    file_path: Path,
    out_dir: Path,
    crop_size: int = 512,
    raster_size: int = 224,
    shift: int = 2 ** 9,
    magic_const: int = 3,
    num_channels: int = 11,
):
    data = np.load(file_path)

    displacement = np.array([[raster_size // 4, raster_size // 2]]) * shift

    traffic_lights = {
        name: set()
        for name in ["green", "yellow", "red"]
    }

    traffic_lights_state = data["traffic_light_state/current/state"]
    traffic_lights_id = data["traffic_light_state/current/id"]
    traffic_lights_mask = data["traffic_light_state/current/valid"]

    traffic_lights_state = traffic_lights_state[traffic_lights_mask]
    traffic_lights_id = traffic_lights_id[traffic_lights_mask]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/waymo_open_dataset_motion_v_1_1_0/preprocessed_npz")
    parser.add_argument("--out-dir", type=str, default="data/waymo_open_dataset_motion_v_1_1_0/rasterized")
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    for split in ["traning", "validation"]:
        print("split", split)

        split_out_dir = out_dir / split
        if not split_out_dir.exists():
            split_out_dir.mkdir(parents=True)

        for file_path in tqdm((data_dir / split).glob("*.npz")):
            rasterize(file_path, split_out_dir)


if __name__ == "__main__":
    main()
